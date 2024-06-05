from Data_Utils import *

def test(model, test_dir, test_name, device, block_size):

    ext = {'/*.jpg', '/*.png', '/*.tif'}
    test_set = os.path.join(test_dir, test_name)
    file_paths = []
    for img_type in ext:
        file_paths = file_paths + glob.glob(test_set + img_type)

    Img_Num = len(file_paths)
    PSNR_ALL = np.zeros([1, Img_Num], dtype=np.float32)
    SSIM_ALL = np.zeros([1, Img_Num], dtype=np.float32)
    TIME_ALL = np.zeros([1, Img_Num], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for img_no in range(Img_Num):

            Img_name = file_paths[img_no]

            Img = cv2.imread(Img_name, 1)
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)

            Iorg_y = Img_yuv[:, :, 0]
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py2(Iorg_y, block_size)


            Img_out = Ipad / 255.0
            batch_x = torch.from_numpy(Img_out)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.unsqueeze(0).unsqueeze(0)
            batch_x = batch_x.to(device)


            start = time()
            x_output, Phi = model(batch_x)
            end = time()

            x_output = x_output.squeeze(0).squeeze(0)
            Prediction_value = x_output.cpu().data.numpy()
            x_rec = np.clip(Prediction_value[:row, :col], 0, 1)

            rec_PSNR = psnr(x_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(x_rec * 255, Iorg.astype(np.float64), data_range=255)

            test_name_split = os.path.split(Img_name)
            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
                img_no, Img_Num, test_name_split[1], (end - start), rec_PSNR, rec_SSIM))

            PSNR_ALL[0, img_no] = rec_PSNR
            SSIM_ALL[0, img_no] = rec_SSIM
            TIME_ALL[0, img_no] = end - start
            del x_output

        return np.mean(PSNR_ALL), np.mean(SSIM_ALL), np.mean(TIME_ALL)