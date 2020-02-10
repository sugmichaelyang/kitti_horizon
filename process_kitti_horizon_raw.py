import argparse
import glob
import os
import pickle
from kitti_horizon_raw import KITTIHorizonRaw

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--raw_path', default='/data/scene_understanding/KITTI/rawdata', type=str,
                        help='path to KITTI rawdata')
    parser.add_argument('--target_path', default='/data/kluger/tmp/kitti_horizon_test', type=str,
                        help='path to save processed data')
    parser.add_argument('--image_scale', default=0.5, type=float,
                        help='image scaling factor')
    args = parser.parse_args()

    dataset = KITTIHorizonRaw(dataset_path=args.raw_path, img_scale=args.image_scale)

    dates = [
        '2011_09_26',
        '2011_09_28',
        '2011_09_29',
        '2011_09_30',
        '2011_10_03'
    ]

    for date in dates:

        date_dir = args.raw_path + "/" + date
        drive_dirs = glob.glob(date_dir + "/*sync")
        drive_dirs.sort()

        drives = []
        for drive_dir in drive_dirs:
            drive = drive_dir.split("_")[-2]
            drives.append(drive)

        for drive_id in drives:

            target_dir = os.path.join(args.target_path, "%s/%s" % (date, drive_id))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            drive = dataset.get_drive(date, drive_id)

            num_images = len(drive)

            for idx, image in enumerate(iter(drive.rgb)):
                data = dataset.process_single_image(drive, image, idx)

                pickle_file = target_dir + "/%06d.pkl" % idx

                print(pickle_file)

                with open(pickle_file, 'wb') as f:
                    pickle.dump(data, f, -1)



