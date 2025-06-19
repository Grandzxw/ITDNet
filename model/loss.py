import torch
import torch.nn as nn

def quadruplet_loss(q_vec, pos_vec, neg_vec, other_neg, m1, m2):
    pos_dis = ((q_vec - pos_vec)**2).sum(dim=1)
    neg_dis = ((q_vec - neg_vec)**2).sum(dim=1)
    other_dis = ((neg_vec - other_neg)**2).sum(dim=1)
    triplet_loss = m1 + pos_dis - neg_dis
    triplet_loss = triplet_loss.clamp(min=0.0)
    second_loss = m2 + pos_dis - other_dis
    second_loss = second_loss.clamp(min=0.0)
    sum_loss = triplet_loss + second_loss
    mask = (sum_loss > 0)
    return pos_dis, neg_dis, other_dis, torch.sum(
        sum_loss) / (torch.sum(mask) + 1e-6)
    # return torch.mean(triplet_loss + second_loss)

def triplet_margin_loss(query, positive, negative, margin=0.1):
    distance_positive = torch.norm(query - positive, dim=1, p=2)**2
    distance_negative = torch.norm(query - negative, dim=1, p=2)**2
    losses = torch.relu(distance_positive - distance_negative + margin)
    loss = torch.mean(losses)
    return loss


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce_loss_fn = nn.BCELoss()
    
    def forward(self, real_output, fake_output):
        real_loss = self.bce_loss_fn(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss_fn(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2



if __name__ == "__main__":
    pass
