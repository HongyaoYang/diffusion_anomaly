import os
import nibabel as nibabel
import torch


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        """
        BraTS 2D slice dataset.

        Expected training structure:

        data/brats_2d_z27_75/
        └── training_2d_p001/
            └── z027/
                ├── brats_train_001_flair_027_w.nii.gz
                ├── brats_train_001_t1_027_w.nii.gz
                ├── brats_train_001_t1ce_027_w.nii.gz
                ├── brats_train_001_t2_027_w.nii.gz
                └── brats_train_001_seg_027_w.nii.gz

        Each zXXX folder is one 2D sample.
        Model input shape after padding:
            [4, 256, 256]
        """

        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag

        if test_flag:
            self.seqtypes = ["t1", "t1ce", "t2", "flair"]
        else:
            self.seqtypes = ["t1", "t1ce", "t2", "flair", "seg"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # Only use leaf folders such as:
            # training_2d_p001/z027/
            if dirs:
                continue

            nii_files = sorted(
                [
                    f for f in files
                    if f.endswith(".nii") or f.endswith(".nii.gz")
                ]
            )

            if len(nii_files) == 0:
                continue

            datapoint = {}

            for f in nii_files:
                parts = f.split("_")

                # Expected filename:
                # brats_train_001_flair_027_w.nii.gz
                #   0     1     2     3    4   5
                if len(parts) < 6:
                    continue

                seqtype = parts[3]

                if seqtype in self.seqtypes_set:
                    datapoint[seqtype] = os.path.join(root, f)

            if set(datapoint.keys()) == self.seqtypes_set:
                slice_id = os.path.basename(root)  # z027
                patient_id = os.path.basename(os.path.dirname(root))  # training_2d_p001

                datapoint["number"] = f"{patient_id}_{slice_id}"

                self.database.append(datapoint)

            else:
                if len(datapoint) > 0:
                    print(
                        f"Skipping incomplete folder: {root}, "
                        f"found={sorted(datapoint.keys())}, "
                        f"expected={sorted(self.seqtypes_set)}"
                    )

        if len(self.database) == 0:
            raise RuntimeError(
                f"No valid BraTS 2D samples found in {self.directory}. "
                f"Check your folder structure and filenames."
            )

        print(f"Loaded {len(self.database)} BraTS 2D samples from {self.directory}")

    def __getitem__(self, index):
        filedict = self.database[index]
        number = filedict.get("number", "unknown")

        out = []

        for seqtype in self.seqtypes:
            img = nibabel.load(filedict[seqtype])
            arr = img.get_fdata()
            out.append(torch.tensor(arr, dtype=torch.float32))

        out = torch.stack(out)

        out_dict = {}

        if self.test_flag:
            # Test mode: only 4 MRI modalities.
            image = torch.zeros(4, 256, 256, dtype=torch.float32)
            image[:, 8:-8, 8:-8] = out

            # If you later build a test loader with labels, modify this part.
            label = torch.zeros(1, 240, 240, dtype=torch.float32)
            weak_label = 0
            out_dict["y"] = weak_label

        else:
            # Training mode:
            # out[0:4] = t1, t1ce, t2, flair
            # out[4]   = seg
            image = torch.zeros(4, 256, 256, dtype=torch.float32)
            image[:, 8:-8, 8:-8] = out[:-1, ...]

            label = out[-1, ...][None, ...]
            weak_label = 1 if label.max() > 0 else 0

            out_dict["y"] = weak_label

        return image, out_dict, weak_label, label, number

    def __len__(self):
        return len(self.database)