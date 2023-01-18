import os, sys
from libs import *

def client_fit_fn(
    fit_loaders, num_epochs, 
    client_model, 
    optimizer, 
    device = torch.device("cpu"), 
):
    print("\nStart Client Fitting ...\n" + " = "*16)
    client_model = client_model.to(device)

    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

        client_model.train()
        running_loss, running_corrects,  = 0.0, 0.0, 
        for images, labels in tqdm.tqdm(fit_loaders["fit"]):
            images, labels = images.to(device), labels.to(device)

            logits = client_model(images.float())
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
        fit_loss, fit_accuracy,  = running_loss/len(fit_loaders["fit"].dataset), running_corrects/len(fit_loaders["fit"].dataset), 
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "fit", 
            fit_loss, fit_accuracy, 
        ))

        with torch.no_grad():
            client_model.eval()
            running_loss, running_corrects,  = 0.0, 0.0, 
            for images, labels in tqdm.tqdm(fit_loaders["evaluate"]):
                images, labels = images.to(device), labels.to(device)

                logits = client_model(images.float())
                loss = F.cross_entropy(logits, labels)

                running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
        evaluate_loss, evaluate_accuracy,  = running_loss/len(fit_loaders["evaluate"].dataset), running_corrects/len(fit_loaders["evaluate"].dataset), 
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "evaluate", 
            evaluate_loss, evaluate_accuracy, 
        ))

    print("\nFinish Client Fitting ...\n" + " = "*16)
    return {
        "fit_loss":fit_loss, "fit_accuracy":fit_accuracy, 
        "evaluate_loss":evaluate_loss, "evaluate_accuracy":evaluate_accuracy, 
    }

def server_test_fn(
    test_loader, 
    server_model, 
    device = torch.device("cpu"), 
):
    print("\nStart Server Testing ...\n" + " = "*16)
    server_model = server_model.to(device)

    with torch.no_grad():
        server_model.eval()
        running_loss, running_corrects,  = 0.0, 0.0, 
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            logits = server_model(images.float())
            loss = F.cross_entropy(logits, labels)

            running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
    test_loss, test_accuracy,  = running_loss/len(test_loader.dataset), running_corrects/len(test_loader.dataset), 
    print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
        "test", 
        test_loss, test_accuracy, 
    ))

    print("\nFinish Server Testing ...\n" + " = "*16)
    return {
        "test_loss":test_loss, "test_accuracy":test_accuracy, 
    }