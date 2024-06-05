from Data_Utils import *

def train(rand_loader, model, optimizer, device, epoch, end_epoch):
    epoch_total_loss = 0
    num = len(rand_loader)
    epoch_discrpancy_loss = 0
    model.train()
    for data in rand_loader:
        batch_x = data.to(device)
        [x_output, phi] = model(batch_x)
        sys_phi = torch.mm(phi, phi.t())
        # Compute discrepancy loss
        loss_discrpancy = torch.sum(torch.pow(x_output - batch_x, 2))
        # Compute orthogonal loss
        gamma = torch.Tensor([0.2]).to(device)
        loss_phi = torch.sum(torch.pow(sys_phi - torch.eye(sys_phi.size(0)).to(device), 2))
        # Compute total loss
        loss_total = loss_discrpancy + gamma * loss_phi
        # loss backward
        optimizer.zero_grad()  # 初始化梯度信息
        loss_total.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # print output
        epoch_discrpancy_loss = epoch_discrpancy_loss + loss_discrpancy.item()
        epoch_total_loss = epoch_total_loss + loss_total.item()
        output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy loss: %.4f, loss_Phi: %.4f\n" % (
            epoch, end_epoch, loss_total.item(), loss_discrpancy.item(), loss_phi.item())
        print(output_data)

    return (epoch_total_loss / num), (epoch_discrpancy_loss / num), loss_phi.item()
