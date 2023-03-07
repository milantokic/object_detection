import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=10):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 11:15], target[...,11:15])
        iou_b2 = intersection_over_union(predictions[..., 16:20], target[...,11:15])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        # exists_box je binarna promenljiva koja odredjuje da li se nalazi objekat ili ne
        exists_box = target[..., 10].unsqueeze(3)

        # Racunanje gubitka za koordinate kutije

        # Ako se nalazi objekat u celiji onda se racuna
        box_predictions = exists_box * (
            (
                    best_box * predictions[..., 16:20]
                    + (1 - best_box) * predictions[..., 11:15]
            )
        )
        # Ako se nalazi objekat onda je cilj za koordinate kutije dat sa box_targets
        box_targets = exists_box * target[..., 11:15]
        # box_predictions[...,2:4] su koordinate za visinu i sirinu kutije ali one koja je ocenjena
        # kako je u formuli za funkciju gubitka koren razlike, transformisemo visinu i sirinu
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        #Box loss racuna mse od kvadratnih korena ocekivane i predvidjene vrednosti za koordinate kutije
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # Racunanje dela funkcije gubitka koji je zaduzen za racunanje verovatnoce kada se u celiji nalazi objekat

        # pred_box odredjuje koja kutija ima najvecu verovatnocu da sadrzi objekat u celiji
        pred_box = (
                best_box * predictions[..., 15:16] + (1 - best_box) * predictions[..., 10:11]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,10:11])
        )

        # Racunanje funkcije gubitka u celiji gde nema objekta

        # Ovo je bitno zato sto zelimo da kaznimo model kada odredi da je velika verovatnoca da postoji objekat u celiji
        # gde ga nema

        # (1 - exists_box) je ona celija u kojoj nema objekta
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 10:11], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 10:11], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 15:16], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 10:11], start_dim=1)
        )

        #Racunanje funkcije gubitka za odredjivanje klase objekta

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[...,:10], end_dim=-2),
            torch.flatten(exists_box * target[..., :10], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss + # Prva dva reda funkcije gubitka u radu
             object_loss # Treci red u radu
            + self.lambda_noobj * no_object_loss # Cetvrti red u radu
            + class_loss # Peti red u radu
        )

        return loss
