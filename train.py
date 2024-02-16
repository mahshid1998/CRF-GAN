import torch
import os
import json
import argparse

from torch import nn
from torch import optim
from tensorboardX import SummaryWriter

from utils import trim_state_dict_name, inf_train_gen
from volume_dataset import Volume_Dataset
from torch.backends import cudnn
from torch.nn import functional as F


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Classifier Training')
parser.add_argument('--batch-size', default=4, type=int,
                    help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--img-size', default=256, type=int,
                    help='size of training images (default: 256, can be 128 or 256)')
parser.add_argument('--num-iter', default=80000, type=int,
                    help='number of iteration for training (default: 80000)')
parser.add_argument('--log-iter', default=20, type=int,
                    help='number of iteration between logging (default: 20)')

parser.add_argument('--lr-d', default=0.0004, type=float,
                    help='learning rate for the discriminator')
parser.add_argument('--data-dir', type=str,
                    help='path to the preprocessed data folder')
parser.add_argument('--exp-name', default='CRF_GAN_run1', type=str,
                    help='name of the experiment')
parser.add_argument('--fold', default=0, type=int,
                    help='fold number for cross validation')

# configs for conditional generation
parser.add_argument('--lambda_class', default=0.1, type=float,
                    help='weights for the auxiliary classifier loss')
parser.add_argument('--num_class', default=0, type=int,
                    help='number of class for auxiliary classifier (0 if unconditional)')


def main():
    # Configuration
    args = parser.parse_args()
    trainset = Volume_Dataset(data_dir=args.data_dir, fold=args.fold, num_class=args.num_class)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=False, num_workers=args.workers)
    gen_load = inf_train_gen(train_loader)

    from models.Model_HA_GAN_128 import Discriminator, Generator
        
    G = Generator(mode='eval', latent_dim=1024, num_class=args.num_class).cuda()
    D = Discriminator(num_class=args.num_class).cuda()

    d_optimizer = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.999), eps=1e-8)

    # Resume from a previous checkpoint
    ckpt_path = './checkpoint/'+args.exp_name+'/G_iter80000.pth'
    ckpt = torch.load(ckpt_path, map_location='cuda')
    ckpt['model'] = trim_state_dict_name(ckpt['model'])
    G.load_state_dict(ckpt['model'])
    # g_optimizer.load_state_dict(ckpt['optimizer'])

    '''
        ckpt_path = './checkpoint/'+args.exp_name+'/D_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
    '''

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

    D.train()
    G.eval()

    # loss_f = nn.BCEWithLogitsLoss()
    # loss_mse = nn.L1Loss()

    summary_writer = SummaryWriter("./checkpoint/"+args.exp_name)
    # save configurations to a dictionary
    with open(os.path.join("./checkpoint/"+args.exp_name, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    for p in D.parameters():  
        p.requires_grad = True
    for p in G.parameters():
        p.requires_grad = False

    print("I am classifier")
    for iteration in range(0, args.num_iter):
        ###############################################
        # Train Discriminator (D^H)
        ###############################################

        # loading image, cropping, down sampling
        real_images, class_label = gen_load.__next__()
        D.zero_grad()

        real_images = real_images.float().cuda()
        print("real images: ", real_images.shape)
        # randomly select a high-res sub-volume from real image
        # crop_idx = np.random.randint(0, args.img_size*7/8+1)  # 256 * 7/8 + 1
        # real_images_crop = real_images[:, :, crop_idx:crop_idx+args.img_size//8, :, :]

        # print("conditional D")
        class_label_onehot = F.one_hot(class_label, num_classes=args.num_class)
        class_label = class_label.long().cuda()
        class_label_onehot = class_label_onehot.float().cuda()

        # y_real_pred, y_real_class = D(real_images_crop, real_images_small, crop_idx)
        y_real_class = D(real_images)
        # GAN loss + auxiliary classifier loss
        d_real_loss = F.cross_entropy(y_real_class, class_label)

        # random generation
        noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
        fake_images = G(noise, class_label=class_label_onehot)
        print("fake images: ", fake_images.shape)

        y_fake_class = D(fake_images)
        d_fake_loss = F.cross_entropy(y_fake_class, class_label)

        d_loss = (d_real_loss + d_fake_loss) / 2.
        d_loss.backward()
        d_optimizer.step()

        # Logging
        if iteration % args.log_iter == 0:
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
            summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
            print(iteration, "iter")
            print('[{}/{}]'.format(iteration, args.num_iter),
                  'D_real: {:<8.3}'.format(d_real_loss.item()),
                  'D_fake: {:<8.3}'.format(d_fake_loss.item()), )

        if iteration > 30000 and (iteration+1) % 5000 == 0:
            torch.save({'model': D.state_dict(), 'optimizer': d_optimizer.state_dict()},
                       './checkpoint/'+args.exp_name+'/D_iter'+str(iteration+1)+'.pth')


if __name__ == '__main__':
    main()
