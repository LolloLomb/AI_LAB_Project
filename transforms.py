from torchvision import transforms

# TRANSFORMS

train_transforms = transforms.Compose(
   [
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ],
)
test_transforms = transforms.Compose(
   [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)