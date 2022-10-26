import os
import argparse
import torch
import time
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from voxel_data import VoxelData

# basic AutoEncoder, include encoder and decoder.
# model input is [batch, feature]
# use in data compressing

model_save = "./Model_save"
pic_save = "./Pic_save/AutoEncoder/"
log_save = "./experiment"
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists("./Pic_save"):
    os.mkdir("./Pic_save")
if not os.path.exists(pic_save):
    os.mkdir(pic_save)
if not os.path.exists(log_save):
    os.mkdir(log_save)

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

class Encoder(nn.Module):
    """The encoder for VAE"""
    
    def __init__(self, input_dim, hidden_dim, fc_dim, latent_dim):
        super().__init__()
        
        convs = []
        prev_dim = input_dim


        convs.append(nn.Sequential(
            nn.Conv3d(prev_dim, 16, kernel_size=3, stride=1, padding=1), # 20
            nn.ReLU()
        ))
        convs.append(nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        convs.append(nn.Sequential(
            nn.MaxPool3d(2,2)             # 10
        ))
        convs.append(nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=2, stride=1),
            nn.ReLU()
        ))
        convs.append(nn.Sequential(
            nn.Conv3d(64, hidden_dim, kernel_size=2, stride=1),
            nn.ReLU()
        ))
        convs.append(nn.Sequential(
            nn.MaxPool3d(2,2)             # 4
        ))

        self.convs = nn.Sequential(*convs)
        
        prev_dim = 4 ** 3 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, fc_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_log_var = nn.Linear(fc_dim, latent_dim)
                    
    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    """The decoder for VAE"""
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        
        fc_dim = 4 ** 3 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU()
        )
        self.conv_size = 4
        
        de_convs = []
        prev_dim = hidden_dim

        de_convs.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
        ))
        de_convs.append(nn.Sequential(
            nn.ConvTranspose3d(prev_dim, 64, kernel_size=2, stride=1),
            nn.ReLU()
        ))
        de_convs.append(nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=1),
            nn.ReLU()
        ))
        de_convs.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
        ))
        de_convs.append(nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        de_convs.append(nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        prev_dim = 16

        self.de_convs = nn.Sequential(*de_convs)
        self.pred_layer = nn.Sequential(
            nn.Conv3d(prev_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.size(0), -1, self.conv_size, self.conv_size, self.conv_size)
        #print(x.shape)
        x = self.de_convs(x)
        x = self.pred_layer(x)
        return x

class VAE(nn.Module):
    """VAE"""
    
    def __init__(self, input_dim, hidden_dim, fc_dim, latent_dim):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, fc_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var
    
    def compute_loss(self, x, recon, mu, log_var):
        """compute loss of VAE"""
        
        # KL loss
        kl_loss = (0.5*(log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        
        # recon loss
        recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum([1, 2, 3]).mean()
        
        return kl_loss + recon_loss


def psnr(img1, img2):
   mse = torch.mean( (img1 - img2) ** 2,dim=[1, 2, 3] )
   #if mse < 1.0e-10:
   #   return 100
   PIXEL_MAX = 1
   return 20 * torch.mean(torch.log10(PIXEL_MAX / torch.sqrt(mse)))

def train(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f_log = IOStream(f'experiment/{timestr}.txt')
    f_log.write(str(args))
    # setting
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    #data = datasets.MNIST('data', train=True, download=True, transform=transform)
    #train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    train_loader = DataLoader(VoxelData(), num_workers=8,batch_size=args.batch_size, shuffle=True, drop_last=True)

    # create model
    model = VAE(9, 128, 128, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # start train
    best_loss = 1e10
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        datasize = 0
        for idx, img in enumerate(train_loader):
            b_size = img.size()[0]
            img = img.permute(0, 4, 1, 2, 3).to(device)
            optimizer.zero_grad()

            prediction, mu, log_var = model(img)
            loss = model.compute_loss(img, prediction, mu, log_var)
            train_loss += loss * b_size
            datasize += b_size

            loss.backward()
            optimizer.step()

            # ---------- print log ---------------
            if idx % args.log_interval == 0:
                f_log.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f} PSNR: {:.6f}'.format(
                    epoch, idx, len(train_loader), 100. * idx / len(train_loader), loss, psnr(img,prediction)))

        train_loss /= datasize
        f_log.write('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

        # ------------ save model ---------------
        if best_loss > train_loss:
            best_loss = train_loss
            try:
                torch.save(model.module.state_dict(), os.path.join(model_save, "AutoEncoder.pkl"))
            except:
                torch.save(model.state_dict(), os.path.join(model_save, "AutoEncoder.pkl"))
            print("Save model!")

        # ----------- save image ---------------
        # if epoch % 5 == 0:
        #     img_save(img, os.path.join(pic_save, "img_%d.png" % (epoch)))
        #     img_save(img_noise, os.path.join(pic_save, "noise_%d.png" % (epoch)))
        #     img_save(prediction, os.path.join(pic_save, "pred_%d.png" % (epoch)))

def img_save(image, url):
    image = image.view(image.shape[0], 1, 28, 28).cpu().data
    save_image(image, url, nrow=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 50)')
    parser.add_argument('--ckpt', type=str, default='exp', metavar='N',
                        help='log name')
    args = parser.parse_args()
    train(args)