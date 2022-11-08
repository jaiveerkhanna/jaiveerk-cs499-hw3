import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
import json  # to parse the file

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)


def encode_data(data, v2i, seq_len, a2i, t2i):
    # Modified form lecture code's encode data, mine is different because our data is 3 dimensional (2 classses for 1 input)

    n_instructions = len(data[0])
    n_actions = len(a2i)
    n_targets = len(t2i)

    x = np.zeros((n_instructions, seq_len), dtype=np.int32)
    y = np.zeros((n_instructions))
    z = np.zeros((n_instructions), dtype=np.int32)

    # Don't know if i'll need these variables but might be helpful to track this data
    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0

    # WILL DEFINITELY NEED TO REVIEW THIS ITERATION
    for instruction, classes in data[0]:
        # they did this in lecture code but is it really necessary, think we did already earlier
        instruction = preprocess_string(instruction)
        x[idx][0] = v2i["<start>"]  # add start token
        jdx = 1
        for word in instruction.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                # can double check if we want to track unks
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        # DOUBLE CHECK BELOW CODE BASED ON FINAL ITERATION IMPLEMENTAION : Saving the action and target class for the instruction
        y[idx] = a2i[classes[0]]
        z[idx] = t2i[classes[1]]
        idx += 1

    # COPIED print statements from lecture code to help with debugging
    """ print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx) """
    return x, y, z


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    # https://www.geeksforgeeks.org/read-json-file-using-python/
    # Open file and save as a json object
    f = open('lang_to_sem_data.json')
    data = json.load(f)

    # Process the input lines first (create a list of objects that are:  ["input text"],[ ["action classifier"], ["object classifer"]])
    processed_train_data = []
    count = 0
    for episode in data['train']:
        count += 1
        if count == 2:
            break
        # print(episode) --> correctly looking at one episode at a time

        concat_instructions = ""
        concat_classifiers = []
        for step in episode:
            instruction = preprocess_string(step[0])
            action_classifier = preprocess_string(step[1][0])
            target_classifier = preprocess_string(step[1][1])

            concat_instructions += instruction
            concat_instructions += ". "
            concat_classifiers.append((action_classifier, target_classifier))
        # print(concat_instructions)
        # print(concat_classifiers[0])
        processed_train_data.append((concat_instructions, concat_classifiers))
    # print(processed_train_data)

    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(
        processed_train_data)

    # print(vocab_to_index)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(
        processed_train_data)
    # print(actions_to_index)

    # ENCODE DATA:
    train_np_x, train_np_y, train_np_z = encode_data(
        processed_train_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)

    exit()

    train_loader = None
    val_loader = None
    return train_loader, val_loader


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    model = None
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = None
    optimizer = None

    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        output = model(inputs, labels)

        loss = criterion(output.squeeze(), labels[:, 0].long())

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        em = output == labels
        prefix_em = prefix_em(output, labels)
        acc = 0.0

        # logging
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000,
                        help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
