import numpy as np
import os
import pykitti
from skimage import transform


class KITTIHorizonRaw:

    def __init__(self, dataset_path, target_width=1250, target_height=380, img_scale=1.):
        self.basedir = dataset_path
        self.target_width = target_width
        self.target_height = target_height
        self.scale = img_scale

        self.dates = [
            '2011_09_26',
            '2011_09_28',
            '2011_09_29',
            '2011_09_30',
            '2011_10_03'
        ]

    def get_date_ids(self):
        dates = []
        for entry in os.listdir(self.basedir):
            if os.path.isdir(os.path.join(self.basedir, entry)):
                dates += [entry]
        return dates

    def get_drive_ids(self, date):
        drives = []
        date_dir = os.path.join(self.basedir, date)
        for entry in os.listdir(date_dir):
            if os.path.isdir(os.path.join(date_dir, entry)):
                drive = entry.split("_")[-2]
                drives += [drive]
        return drives

    def get_drive(self, date_id, drive_id):
        dataset = pykitti.raw(self.basedir, date_id, drive_id)
        return dataset

    def process_single_image(self, drive, image, idx):

        R_cam_imu = np.matrix(drive.calib.T_cam2_imu[0:3,0:3])
        K = np.matrix(drive.calib.P_rect_20[0:3, 0:3])
        G = np.matrix([[0.], [0.], [1.]])

        orig_image_width = image[0].width

        pad_w = self.target_width - image[0].width
        pad_h = self.target_height - image[0].height

        pad_w1 = int(pad_w / 2)
        pad_w2 = pad_w - pad_w1
        pad_h1 = int(pad_h / 2)
        pad_h2 = pad_h - pad_h1

        padded_image = np.pad(np.array(image[0]), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), 'edge')
        if self.scale < 1.:
            padded_image = transform.rescale(padded_image, (self.scale, self.scale, 1))

        R_imu = np.matrix(drive.oxts[idx].T_w_imu[0:3, 0:3])
        G_imu = R_imu.T * G
        G_cam = R_cam_imu * G_imu

        h = np.array(K.I.T * G_cam).squeeze()

        padded_image = np.transpose(padded_image, [2, 0, 1]).astype(np.float32)

        hp1 = np.cross(h, np.array([1, 0, 0]))
        hp2 = np.cross(h, np.array([1, 0, -orig_image_width]))
        hp1 /= hp1[2]
        hp2 /= hp2[2]

        hp1[0] += pad_w1
        hp2[0] += pad_w1
        hp1[1] += pad_h1
        hp2[1] += pad_h1

        hp1[0:2] *= self.scale
        hp2[0:2] *= self.scale

        offset = (0.5 * (hp1[1] + hp2[1])) / self.target_height - 0.5

        h = np.cross(hp1, hp2)

        angle = np.arctan2(h[0], h[1])
        if angle > np.pi / 2:
            angle -= np.pi
        elif angle < -np.pi / 2:
            angle += np.pi

        h = h / np.linalg.norm(h[0:2])

        data = {'image': padded_image, 'horizon_hom': h, 'horizon_p1': hp1, 'horizon_p2': hp2, 'offset': offset,
                'angle': angle, 'padding': (pad_w1, pad_w2, pad_h1, pad_h2), 'G': G_cam, 'K': K}
        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='Shows images from the raw KITTI dataset with horizons')
    parser.add_argument('--path', default='/tnt/data/scene_understanding/KITTI/rawdata', type=str,
                        help='path to KITTI rawdata')
    parser.add_argument('--date', default=None, type=str, help='')
    parser.add_argument('--drive', default=None, type=str, help='')
    args = parser.parse_args()

    dataset = KITTIHorizonRaw(dataset_path=args.path, img_scale=0.5)

    if args.date is None:
        all_dates = dataset.get_date_ids()
        print("available dates:")
        for date in all_dates:
            print(date)
        exit(0)

    if args.drive is None:
        all_drives = dataset.get_drive_ids(args.date)
        print("available drives:")
        for drive in all_drives:
            print(drive)
        exit(0)

    drive = dataset.get_drive(args.date, args.drive)

    num_images = len(drive)

    for idx, image in enumerate(iter(drive.rgb)):
        data = dataset.process_single_image(drive, image, idx)
        processed_image = np.transpose(data['image'], [1, 2, 0])/255.
        hp1 = data['horizon_p1']
        hp2 = data['horizon_p2']

        plt.figure()
        plt.imshow(processed_image)
        plt.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], 'b-', lw=5)
        plt.show()



