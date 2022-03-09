import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset


def train_model(model, dloader, crit, optim, epoch, total_epochs, device):
  model.train()

  running_loss = 0.0
  total_lengths = 0.0
  total_positives = 0.0

  for i, elemen in enumerate(dloader):
    optim.zero_grad()
    inputs, labels = elemen[0].to(device), elemen[1].to(device)

    out = model(inputs)
    loss = crit(out, labels)
    loss.backward()
    optim.step()

    with torch.no_grad():
      soft_out = F.softmax(out, dim=1)
      probs, preds = soft_out.topk(1, dim=1)
      batch_length = len(labels)
      batch_positives = (labels == preds.view(-1)).sum()

      total_lengths+=batch_length
      total_positives+=batch_positives.item()
        
    running_loss+=loss.item()
  
  acc = total_positives/total_lengths
  print(f'[epoch: {epoch+1}/{total_epochs}] loss: {running_loss/len(dloader):.4f} acc:{acc}')


def eval_model(model, dloader, crit, device):
  model.eval()
  
  running_loss = 0.0
  running_positives = 0.0
  running_lengths = 0.0

  with torch.no_grad():

    for i, elemen in enumerate(dloader):
      inputs, labels = elemen[0].to(device), elemen[1].to(device)

      out = model(inputs)
      loss = crit(out, labels)
      soft_out = F.softmax(out, dim=1)
      probs, preds = soft_out.topk(1, dim=1)

      batch_length=len(labels)
      batch_positives=(labels == preds.view(-1)).sum()

      running_lengths+=batch_length
      running_positives+=batch_positives.item()    

      running_loss+=loss.item()
    acc = running_positives/running_lengths
    print('-'*90)
    print(f'[Test eval] loss: {running_loss/len(dloader):.4f} acc: {acc:.8f}')
    print('-'*90)


class ViTDsetPytorch(Dataset):
  '''
  Designed to transform images from a pytorch dataset (e.g. CIFAR10)
  to img patches for a vision transformer model
  '''
  def __init__(self, dset, patch_size, img_size=72):
    self.dset = dset
    self.patch_size = patch_size
    self.img_size = img_size
    self.num_patches = (img_size//patch_size)**2

  def __len__(self):
    return len(self.dset)
  
  def __getitem__(self, idx):
    img, label = self.dset[idx]
    img = transforms.Resize(self.img_size)(img)

    img_patches = self.get_patches(img, self.patch_size)
    img_patches = img_patches.permute(1, 2, 3, 0) #moving channels from first to last
    img_patches = img_patches.contiguous().view(img_patches.shape[0], -1)

    return img_patches, label

  @staticmethod
  def get_patches(img, patch_size):
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size)
    return patches #[c, n_p, h, w]