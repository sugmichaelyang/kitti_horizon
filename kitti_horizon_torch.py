from torch.utils.data import Dataset
import csv
import numpy as np
import pickle
import cv2
from PIL import Image


class KITTIHorizon(Dataset):

    def __init__(self, csv_file, root_dir, seq_length, augmentation=True, return_info=False,
                 fill_up=True, transform=None, single_sequence=None,
                 max_shift=10., max_rotation=2., padding=0):

        self.seq_length = seq_length
        self.transform = transform
        self.max_shift = max_shift
        self.max_rotation = max_rotation

        self.num_images = 0

        self.sequences = []

        if csv_file is None:
            date = single_sequence[0]
            drive = single_sequence[1]
            start = single_sequence[2]
            end = single_sequence[3]
            total_length = end-start
            self.sequences.append((date, drive, (0, total_length), start))

        else:
            print("csv file: ", csv_file)

            with open(csv_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')

                for row in reader:
                    date = row[0]
                    drive = row[1]
                    total_length = int(row[2])
                    start_frame = int(row[3])
                    self.num_images += total_length

                    if total_length <= self.seq_length:
                        self.sequences.append((date, drive, (0, total_length), start_frame))
                    else:

                        start_range = (range(0, 0+total_length-self.seq_length+1, self.seq_length))
                        stop_range = (range(self.seq_length+padding, 0+total_length+1, self.seq_length))

                        for frames in zip(start_range, stop_range):
                            self.sequences.append((date, drive, frames, start_frame))

                        trailing = total_length % self.seq_length
                        if trailing > 0:
                            self.sequences.append((date, drive, (total_length-trailing, total_length), start_frame))

        print(self.num_images, " images")

        self.root_dir = root_dir
        self.augmentation = augmentation
        self.return_info = return_info

        self.fill_up = fill_up
        self.im_width = None
        self.im_height = None
        self.padding = padding

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        date = self.sequences[idx][0]
        drive = self.sequences[idx][1]
        frames = self.sequences[idx][2]
        start_frame = self.sequences[idx][3]

        frame_list = list(range(frames[0],frames[1]))

        dataset = [((self.root_dir + "/" + date + "/" + drive + "/%06d.pkl" % (idx+start_frame)), idx) for idx in frame_list]

        if self.fill_up:
            seq_length = self.seq_length + self.padding
        else:
            seq_length = len(dataset)

        filename = dataset[0][0]
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            image = np.transpose(data['image'], [1, 2, 0])

            self.im_width = image.shape[1]
            self.im_height = image.shape[0]

        images = np.zeros((seq_length, 3, self.im_height, self.im_width)).astype(np.float32)
        offsets = np.zeros((seq_length, 1)).astype(np.float32)
        angles = np.zeros((seq_length, 1)).astype(np.float32)
        Gs = np.zeros((seq_length, 3)).astype(np.float32)

        if self.augmentation:
            rotation = np.random.uniform(-self.max_rotation, self.max_rotation)

            shift = (np.random.uniform(-self.max_shift, self.max_shift),
                     np.random.uniform(-self.max_shift, self.max_shift), 0)

            rot = -rotation / 180. * np.pi
            Tf = np.matrix([[1, 0, -self.im_width / 2.], [0, 1, -self.im_height / 2.], [0, 0, 1]])
            Tb = np.matrix([[1, 0, self.im_width / 2.], [0, 1, self.im_height / 2.], [0, 0, 1]])
            Rt = Tb * np.matrix(
                [[np.cos(rot), -np.sin(rot), -shift[0]], [np.sin(rot), np.cos(rot), -shift[1]], [0, 0, 1]]) * Tf

        for i, (filename, pidx) in enumerate(dataset):

            if pidx < 0:
                continue

            with open(filename, 'rb') as fp:
                data = pickle.load(fp)

            image = np.transpose(data['image'], [1, 2, 0])

            image_width = image.shape[1]

            h = data['horizon_hom']

            if self.augmentation:

                h = np.array(Rt.I.T * np.matrix(h).T).squeeze()

                angle = np.arctan2(h[0], h[1])
                if angle > np.pi / 2:
                    angle -= np.pi
                elif angle < -np.pi / 2:
                    angle += np.pi

                M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), rotation, 1)
                M[0,2] += shift[0]
                M[1,2] += -shift[1]
                image = cv2.warpAffine(image, M, (0, 0), borderMode=cv2.BORDER_REPLICATE)

                if self.augmentation and np.random.uniform(0., 1.) > 0.5:
                    image = cv2.flip(image, 1)
                    angle *= -1

            else:
                offset = data['offset']
                angle = data['angle']

            hp1 = np.cross(h, np.array([1, 0, 0]))
            hp2 = np.cross(h, np.array([1, 0, -image_width]))
            hp1 /= hp1[2]
            hp2 /= hp2[2]

            offset = (0.5 * (hp1[1] + hp2[1])) / self.im_height - 0.5

            if self.transform is not None:
                image = self.transform(Image.fromarray((image*255.).astype('uint8')))
            else:
                image = np.transpose(image, [2, 0, 1])

            images[i,:,:,:] = image

            if i >= 0:
                offsets[i] = offset
                angles[i] = angle

                if self.return_info:
                    Gs[i, :] = data['G'].squeeze()

        if dataset[0][1] < 0:
            for i in range(-dataset[0][1]):
                images[i,:,:,:] = images[-dataset[0][1],:,:,:]

        if self.fill_up:
            start = len(dataset)
            for i in range(start, self.seq_length):
                images[i,:,:,:] = images[i-1,:,:,:]
                offsets[i] = offsets[i-1]
                angles[i] = angles[i-1]

        sample = {'images': images, 'offsets': offsets, 'angles': angles}

        if self.return_info:
            sample['date'] = date
            sample['drive'] = drive
            sample['start'] = frames[0]
            sample['K'] = np.array(data['K'])
            sample['scale'] = data['scale']
            sample['padding'] = data['padding']
            sample['G'] = Gs

        return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='Shows images from the preprocessed dataset')
    parser.add_argument('--path', default='/tnt/data/kluger/tmp/kitti_horizon_test', type=str,
                        help='path to preprocessed KITTI horizon data')
    parser.add_argument('--idx', default=None, type=int, help='image index to start with')
    parser.add_argument('--set', default="val", type=str, help='train, test, val or all')
    parser.add_argument('--augmentation', dest='augmentation', action='store_true', help='enable data augmentation')
    args = parser.parse_args()

    dataset = KITTIHorizon(root_dir=args.path, csv_file="split/%s.csv" % args.set, seq_length=10000,
                           fill_up=False, augmentation=args.augmentation)
    print("dataset size: %d sequences" % len(dataset))

    start_idx = np.random.randint(0, len(dataset)) if args.idx is None else args.idx

    for idx in range(start_idx, len(dataset)):
        frame = 0
        sample = dataset[idx]

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        image = images[frame, :, :, :].transpose((1, 2, 0))
        width = image.shape[1]
        height = image.shape[0]

        offset = offsets[frame].squeeze()
        offset += 0.5
        offset *= height
        angle = angles[frame].squeeze()

        true_mp = np.array([width / 2., offset])
        true_nv = np.array([np.sin(angle), np.cos(angle)])
        true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
        true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
        true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
        true_h1 /= true_h1[2]
        true_h2 /= true_h2[2]

        fig = plt.figure(figsize=(19.75, 6.0))
        plt.suptitle("offset %.1f px, angle %.1f deg" %
                     (offset, angle * 180. / np.pi), family='monospace')

        plt.imshow(image)
        plt.axis('off')
        plt.autoscale(False)
        plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], '-', lw=14, c="#99c000")
        plt.show()
