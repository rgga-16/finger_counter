import torch
from dataset import FingerCountData
from torchvision import transforms
import torch.optim as optim
import argparse
import models
import config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir',type=str,default='./data/train')
    parser.add_argument('--test_dir',type=str,default=None)
    parser.add_argument('--batch_size','-bs',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--weights_dir',type=str,default='./weights')

    return parser.parse_known_args()[0]



def train_step():

    return 

def train(model, train_loader, optimizer, epochs, weights_dir, test_loader=None):

    handtype_criterion = torch.nn.BCELoss().to(device)

    count_criterion = torch.nn.CrossEntropyLoss()


    for ep in range(epochs):

        for i, data in enumerate(train_loader):
            real_images = data[0].to(device)
            real_counts=data[1].to(device) * 1.0
            real_handtypes=data[2].to(device) * 1.0

            optimizer.zero_grad()

            pred_handtypes,pred_counts = model(real_images)
            # pred_handtypes=pred_handtypes.type(torch.float64).to(device)
            # pred_counts=pred_counts.type(torch.float64).to(device)

            handtype_loss = handtype_criterion(pred_handtypes.squeeze(),real_handtypes)
            real_counts = real_counts.to(torch.int64)

            #BUG: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'
            # Fixed by casting real_counts to torch.int64
            counts_loss = count_criterion(pred_counts,real_counts) 

            total_loss = handtype_loss + counts_loss
            total_loss.backward()
            optimizer.step()

            print()




    return 


def main():

    args = get_args()

    batch_size = 32

    trainset = FingerCountData('./data/train',is_train=True)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model = models.CustomClassifier().to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)


    train(model,train_loader,optimizer,epochs=args.epochs,weights_dir=args.weights_dir)



    return 

if __name__=='__main__':
    device = config.device
    main()