import time


def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader):
        start_time = time.time()

        optimizer.zero_grad()
        print(data)
        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
        loss = ((output - target) ** 2).sum().refresh() / batch_size

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]'
                '\tLoss: {:.6f}'
                '\tTime: {:.3f}s'.format(
                    epoch, batch_idx * args.batch_size,
                           len(private_train_loader) * args.batch_size,
                           100. * batch_idx / len(private_train_loader),
                    loss.item(), time.time() - start_time))
