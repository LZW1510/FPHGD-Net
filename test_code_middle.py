from model_file.FPHGD_Net import *
from argparse import ArgumentParser

from Data_Utils import *
######################################################################################
##########################SET PARSER##################################################
######################################################################################
#parser for test detail
parser = ArgumentParser(description="FPHGD-NET")
parser.add_argument('--epoch_num', type=int, default=120, help='epoch number of model')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help="from{1,4,10,25,40,50}")
parser.add_argument('--gpu_list', type=str, default='0', help="gpu_index")
parser.add_argument('--crop_size', type=int, default=96, help="training images crop size")
parser.add_argument('--block_size', type=int, default=32, help="training images block size")
parser.add_argument("--channel", type=int, default=16, help="the network channel")
parser.add_argument("--layer_num", type=int, default=12, help="the network layer")
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--model_name', type=str, default="FPHGD_Net", help="trained or pre-trained model")
parser.add_argument('--design_method', type=str, default="Adamw-Relu", help="the network design")
parser.add_argument('--test_dir', type=str, default='data', help='test data directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()
epoch_num = args.epoch_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
design_method = args.design_method
crop_size = args.crop_size
block_size = args.block_size
model_name = args.model_name
test_dir = args.test_dir
test_name = args.test_name
result_dir = args.result_dir
channel = args.channel
layer_num = args.layer_num
######################################################################################
##########################SET ENVIRONMENT#############################################
######################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################################
##########################SET MODEL###################################################
######################################################################################
model = CS_Reconstruction(channel, cs_ratio * 0.01, layer_num)
model = nn.DataParallel(model)
model = model.to(device)
######################################################################################
############################SET DIR###################################################
######################################################################################
# model dir
model_dir = "./save_model/%s/group_%d_ratio_%d_cropsize_%d_blocksize_%d_method_%s" % (
    model_name, group_num, cs_ratio, crop_size, block_size, design_method)
# test dir
test_dir = os.path.join(test_dir, test_name)
# result dir
result_dir = os.path.join(result_dir, model_name, str(cs_ratio), test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# Load pre-trained model with epoch number
model.load_state_dict(
    torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num), map_location=device), strict=False)
######################################################################################
############################LOAD DATA#################################################
######################################################################################
ext = {'/*.jpg', '/*.png', '/*.tif'}
filepaths = []
for img_type in ext:
    filepaths = filepaths + glob.glob(test_dir + img_type)
######################################################################################
############################INITIAL VALUE#############################################
######################################################################################
Img_Num = len(filepaths)
PSNR_ALL = np.zeros([1, Img_Num], dtype=np.float32)
SSIM_ALL = np.zeros([1, Img_Num], dtype=np.float32)
TIME_ALL = np.zeros([1, Img_Num], dtype=np.float32)
######################################################################################
############################BEGIN TEST################################################
######################################################################################
print("\n**************CS Reconstruction Start***************")
model.eval()
with torch.no_grad():
    model(torch.zeros(1, 1, 256, 256).to(device))
    for img_no in range(Img_Num):
        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:, :, 0]
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y, block_size)
        Img_output = Ipad / 255.

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.unsqueeze(0).unsqueeze(0)
        batch_x = batch_x.to(device)

        start = time()
        x_output, sys_cons = model(batch_x)
        end = time()

        x_output = x_output.squeeze(0).squeeze(0)
        Prediction_value = x_output.cpu().data.numpy()
        X_rec = np.clip(Prediction_value[:row, :col], 0, 1)

        rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)

        test_name_split = os.path.split(imgName)
        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
            img_no, Img_Num, test_name_split[1], (end - start), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:, :, 0] = X_rec * 255
        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
        resultName = "./%s/%s" % (result_dir, test_name_split[1])
        cv2.imwrite("%s_CSratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (
            resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
        del x_output

        PSNR_ALL[0, img_no] = rec_PSNR
        SSIM_ALL[0, img_no] = rec_SSIM
        TIME_ALL[0, img_no] = end - start
print('\n')
print("\n**************CS Reconstruction end***************")
######################################################################################
############################WRITE OUTPUT DATA#########################################
######################################################################################
output_data = "CS ratio is %d, Epoch number of model is %d ,Avg PSNR/SSIM for %s is %.4f/%.4f, time:%.6f\n" % (cs_ratio,
                  epoch_num, test_name, np.mean(PSNR_ALL), np.mean(SSIM_ALL), np.mean(TIME_ALL))
print(output_data)
output_dir = "./log/%s" % (model_name)
output_file_name = "./log/%s/PSNR_SSIM_Results_CS_ISTA_Net_layer_%d_group_%d_ratio.txt" % \
                   (model_name, group_num, cs_ratio)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_file_name, 'a') as f:
    f.write(output_data)
######################################################################################