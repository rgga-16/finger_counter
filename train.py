import torch
from dataset import FingerCountData
import torchvision
from torchvision import transforms
import torch.optim as optim
import argparse
import models
import config
import os 
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir',type=str,default='./data/train')
    parser.add_argument('--test_dir',type=str,default=None)
    parser.add_argument('--batch_size','-bs',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--weights_dir',type=str,default='./weights')

    return parser.parse_known_args()[0]

def train_step():

    return 

def train(model, train_loader, optimizer, epochs, weights_dir, test_loader=None):

    handtype_criterion = torch.nn.BCELoss().to(device)

    count_criterion = torch.nn.CrossEntropyLoss()

    handtypes = [0.0,1.0] # Left=0.0, Right=1.0
    counts = [0.0,1.0,2.0,3.0,4.0,5.0]

    for epoch in range(epochs):
        running_loss=0.0
        for i, data in enumerate(train_loader):
            real_images = data[0].to(device)
            real_counts=data[1].to(device) * 1.0
            real_handtypes=data[2].to(device) * 1.0

            optimizer.zero_grad()

            pred_handtypes,pred_counts = model(real_images)

            handtype_loss = handtype_criterion(pred_handtypes.squeeze(),real_handtypes)
            real_counts = real_counts.to(torch.int64)

            #BUG: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'
            # Fixed by casting real_counts to torch.int64
            counts_loss = count_criterion(pred_counts,real_counts) 

            total_loss = handtype_loss + counts_loss
            total_loss.backward()
            optimizer.step()

            running_loss+=total_loss.item()
            if i % 100 == 99:
                print(f'[{epoch+1}, {i+1:5d}] Train Batch Loss: {running_loss/100:.3f}')
                running_loss=0.0
            
        if test_loader:
            running_loss=0.0

            correct_counts=0; total_counts=0

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    real_images = data[0].to(device)
                    real_counts=data[1].to(device) * 1.0
                    real_handtypes=data[2].to(device) * 1.0

                    pred_handtypes,pred_counts = model(real_images)
                    handtype_loss = handtype_criterion(pred_handtypes.squeeze(),real_handtypes)
                    real_counts = real_counts.to(torch.int64)

                    counts_loss = count_criterion(pred_counts,real_counts) 
                    total_loss = handtype_loss + counts_loss
                    running_loss+=total_loss.item()

                    _,predicted_count = torch.max(pred_counts.data,1)

                    total_counts+= real_counts.size(0)
                    correct_counts+= (predicted_count==real_counts).sum().item()
                    if i % 100 == 99:
                        print(f'[{epoch+1}, {i+1:5d}] Test Batch Loss: {running_loss/100:.3f}')
                        running_loss=0.0
            test_acc = 100 * correct_counts //total_counts
            print(f'[Epoch {epoch+1}] Test Acc: {test_acc}%')
    
    # One last test
    dataiter = iter(test_loader)
    real_images,real_counts,real_handtypes = next(dataiter)
    
    real_images = real_images.to(device)
    real_counts= real_counts.to(device)
    real_handtypes= real_handtypes.to(device) * 1.0
    
    print('GroundTruth: ', ' '.join(f'{counts[real_counts[j]]}' for j in range(4)))

    pred_types, pred_counts = model(real_images)

    _,predicted_counts = torch.max(pred_counts,1)
    print('Predicted: ', ' '.join(f'{counts[predicted_counts[j]]}' for j in range(4)))
    grid = torchvision.utils.make_grid(real_images,normalize=True)
    plt.imshow(grid.cpu().numpy().transpose(1,2,0)); plt.show()

    torch.save(model.state_dict(),os.path.join(args.weights_dir,"weights.pt"))
        

    return 


def main():

    

    batch_size = 32

    trainset = FingerCountData('./data/train',is_train=True)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

    testset = FingerCountData(args.test_dir,is_train=False)
    test_loader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model = models.CustomClassifier().to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)


    train(model,train_loader,optimizer,epochs=args.epochs,weights_dir=args.weights_dir,test_loader=test_loader)



    return 

if __name__=='__main__':
    args = get_args()
    device = config.device
    main()