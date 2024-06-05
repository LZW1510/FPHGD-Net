from model_file.FPHGD_Net import *
from train_utils import *
from test_utils import *

from argparse import ArgumentParser

from Data_Utils import *
######################################################################################
##########################SET PARSER##################################################
######################################################################################
parser = ArgumentParser(description="FPHGD-NET")
#parser for train details
parser.add_argument("--start_epoch", type=int, default=0, help='epoch number of start training')
parser.add_argument("--end_epoch", type=int, default=120, help="epoch number of end training")
parser.add_argument("--initial_learning_rate", type=float, default=2e-4, help="initial learning rate")
parser.add_argument("--end_learning_rate", type=float, default=4e-5, help="end learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="end learning rate")
parser.add_argument("--group_num", type=int, default=1, help="group number for training")
parser.add_argument("--cs_ratio", type=int, default=50, help="from{1,5,10,30,50}")
parser.add_argument("--gpu_list", type=str, default='0', help="gpu_index")
parser.add_argument("--crop_size", type=int, default=96, help="training images crop size")
parser.add_argument("--block_size", type=int, default=32, help="training images block size")
parser.add_argument("--batch_size", type=int, default=16, help="training images batch size")
#parser for data detail
parser.add_argument("--train_data_dir", type=str, default="trainset", help="training data directory")
parser.add_argument("--test_dir", type=str, default='data', help='test data directory')
parser.add_argument("--test_name", type=str, default='Set11', help='name of test set')
#model detail
parser.add_argument("--model_name", type=str, default="FPHGD_Net", help="trained or pre-trained model")
parser.add_argument("--channel", type=int, default=16, help="the network channel")
parser.add_argument("--layer_num", type=int, default=12, help="the network layer")
parser.add_argument("--design_method", type=str, default="Adamw-Relu", help="the network design")


args = parser.parse_args()
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.initial_learning_rate
end_learning_rate = args.end_learning_rate
weight_decay = args.weight_decay
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
design_method = args.design_method
crop_size = args.crop_size
block_size = args.block_size
batch_size = args.batch_size
train_data_dir = args.train_data_dir
model_name = args.model_name
test_dir = args.test_dir
test_name = args.test_name
channel = args.channel
layer_num = args.layer_num
######################################################################################
##########################SET RAMDOM SEED#############################################
######################################################################################
#fix random seed
setup_seed()
######################################################################################
##########################SET ENVIRONMENT#############################################
######################################################################################
#set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照以下顺序排列gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################################
##########################LOAD DATASET##################################################
######################################################################################
#Load CS Training Data
train_set = TrainDatasetFromFolder(train_data_dir, crop_size, block_size)
######################################################################################
##########################SET MODEL###################################################
######################################################################################
#Define model
model = CS_Reconstruction(channel, cs_ratio*0.01, layer_num)
#model = nn.DataParallel(model)
model = model.to(device)
#model parameters_num
total_parameters_num = sum([p.nelement() for p in model.parameters()])
######################################################################################
##########################SET DATASET###################################################
######################################################################################
#load data
if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4,
                             shuffle=True)
######################################################################################
##########################SET DIR###################################################
######################################################################################
#define save_model dir,log dir:
model_dir = "./save_model/%s/group_%d_ratio_%d_cropsize_%d_blocksize_%d_method_%s" % (
    model_name, group_num, cs_ratio, crop_size, block_size, design_method)
log_dir = "./log/%s" % (model_name)
log_file_name = "./log/%s/group_%d_ratio_%d_cropsize_%d_blocksize_%d_method_%s.txt" % (
    model_name, group_num, cs_ratio, crop_size, block_size, design_method)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
######################################################################################
##########################SET OPTIMIZER AND SCHEDULER#################################
######################################################################################
#define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#define learning schduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, end_epoch, eta_min=end_learning_rate, last_epoch=-1)
######################################################################################
##########################LOAD PRETRAIN MODEL#########################################
######################################################################################
#Load pretrain model
if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d_%s.pkl' % (pre_model_dir, start_epoch, design_method)))
    for i in range(0, start_epoch):
        scheduler.step()
######################################################################################
##########################WRITE DETAIL################################################
######################################################################################
log = "start_epoch:%02d, end_epoch:%02d, initial_learning_rate:%.5f, end_learning_rate:%.5f, cs_ratio:%d, gpu_list:%s," \
      "train_data_dir:%s, model_dir:%s, log_dir:%s, design_method:%s, test_name:%s, total_parameter:%d\n" % (start_epoch, end_epoch, learning_rate,
        end_learning_rate, cs_ratio, gpu_list, train_data_dir, model_dir, log_dir, design_method, test_name, total_parameters_num)
with open(log_file_name, 'a') as f:
    f.write(log)
######################################################################################
##########################INITAIL VALUE###############################################
######################################################################################
loss = []
psnr_loss = []
optimizer.zero_grad()
max_psnr = 0
max_ssim = 0
######################################################################################
##########################BEGIN TRAIN#################################################
######################################################################################
print("\n*******************train start*********************\n")
for epoch in range(start_epoch + 1, end_epoch + 1):
    loss_total, loss_discrepancy, loss_phi = train(rand_loader, model, optimizer, device, epoch, end_epoch)
    avg_psnr, avg_ssim, avg_time = test(model, test_dir, test_name, device, block_size)
    scheduler.step()
    #print and save data
    output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy loss: %.4f, Phi Loss: %.4f\n" % (
        epoch, end_epoch, loss_total, loss_discrepancy, loss_phi)
    print(output_data)
    with open(log_file_name, 'a') as f:
        f.write(output_data)
    output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Avg TIME is %.6f, Epoch number of model is %d \n" % (
        cs_ratio, test_name, avg_psnr, avg_ssim, avg_time, epoch)
    print(output_data)
    with open(log_file_name, 'a') as f:
        f.write(output_data)
    #save each epoch loss
    loss.append(loss_total)
    psnr_loss.append(avg_psnr)
    #save model.pkl
    if epoch == end_epoch or avg_psnr > max_psnr or avg_ssim > max_ssim:
        max_psnr = avg_psnr
        max_ssim = avg_ssim
        torch.save(model.state_dict(), './%s/net_params_%d.pkl' % (model_dir, epoch))  # save only the parameters
print("\n*******************train end*********************")
######################################################################################
##########################PLOT PICTURE################################################
######################################################################################
x = range(0, len(loss))
plt.plot(x, loss, 'r')
plt.ylabel("LOSS_TOTAL")
plt.xlabel("epoch")
plt.title("model_name:" + str(model_name) + "cs_ratio=" + str(cs_ratio))
plt.show()
plt.plot(x, psnr_loss, 'b')
plt.ylabel("PSNR")
plt.xlabel("epoch")
plt.title("model_name:" + str(model_name) + "cs_ratio=" + str(cs_ratio))
plt.show()
######################################################################################