import torch
import torch.nn as nn
from torchtext.legacy.data import TabularDataset, Field, BucketIterator, Iterator
import os
from src.utils.tokenizer import smi_tokenizer
from src.transformer import Encoder, Decoder, Seq2Seq


def init_weights(m: nn.Module):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_model(
    folder_out,
    train_path,
    val_path,
    batch_size,
    threshold,
    epochs,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    out = os.path.join(folder_out, f"deepsmiles_model_{batch_size}_3")

    MAX_SEQ_LEN = 120  # or 150 or 200 depending on your data

    SRC = Field(tokenize=list, init_token="<sos>", eos_token="<eos>", lower=False, include_lengths=True, fix_length=MAX_SEQ_LEN)
    TRG = Field(tokenize=list, init_token="<sos>", eos_token="<eos>", lower=False, fix_length=MAX_SEQ_LEN)

    fields = {'ERROR': ('src', SRC), 'STD_SMILES': ('trg', TRG)}

    train_data = TabularDataset(path=train_path, format="csv", fields=fields, skip_header=False)
    val_data = TabularDataset(path=val_path, format="csv", fields=fields, skip_header=False)

    SRC.build_vocab(train_data, max_size=1000)
    TRG.build_vocab(train_data, max_size=1000)

    from torchtext.legacy.data import BucketIterator

    train_iterator, val_iterator = BucketIterator.splits(
        (train_data, val_data),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True,
    )


    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

    # Define encoder and decoder sizes (customize if needed)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    MAX_LEN = 120

    # Instantiate encoder and decoder
    encoder = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, MAX_LEN, device)
    decoder = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, MAX_LEN,  device)

    # Pass to Seq2Seq
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        device=device,
        loader_train=train_iterator,
        loader_valid=val_iterator,
        out=out,
        SRC=SRC,
        TRG=TRG,
        epochs=epochs
    ).to(device)


    torch.save(SRC.vocab, out + "_vocab_src.pth")
    torch.save(TRG.vocab, out + "_vocab_trg.pth")

    model.train_iterator = train_iterator
    model.valid_iterator = val_iterator
    model.out = out
    model.SRC = SRC
    model.TRG = TRG
    model.epochs = epochs

    return model, out, SRC

def train_model(model, out, assess):
    """Apply given weights (& assess performance or train further) or start training new model

    Args:
        model: initialized model
        out: .pkg file with model parameters
        asses: bool 

    Returns:
        model with (new) weights
    """

    if os.path.exists(f"{out}.pkg") and assess:
        print("Assessing performance of existing model")

        model.load_state_dict(torch.load(f=out + ".pkg"))
        (
            valids,
            loss_valid,
            valids_de,
            df_output,
            df_output_de,
            right_molecules,
            complexity,
            unchanged,
            unchanged_de,
        ) = model.evaluate(True)

        print(valids_de)
        print(unchanged_de)

        # log = open('unchanged.log', 'a')
        # info = f'type: comb unchanged: {unchan:.4g} unchanged_drugex: {unchan_de:.4g}'
        # print(info, file=log, flush = True)
        # print(valids_de)
        # print(unchanged_de)

        # print(unchan)
        # print(unchan_de)
        # df_output_de.to_csv(f'{out}_de_new.csv', index = False)

        # error_de = 1 - valids_de / len(drugex_iter.dataset)
        # print(error_de)
        # df_output.to_csv(f'{out}_par.csv', index = False)

    elif os.path.exists(f"{out}.pkg"):
        print("Continue training of existing model")
        # starts from the model after the last epoch, not the best epoch
        model.load_state_dict(torch.load(f=out + "_last.pkg"))
        # need to change how log file names epochs
        model.train_model()
    else:
        print("Start training model")
        model = model.apply(init_weights)
        model.train_model()

    return model


def correct_SMILES(model, out, error_source, device, SRC):
    """Model that is given corrects SMILES and return number of correct ouputs and dataframe containing all outputs
    Args:
        model: initialized model
        out: .pkg file with model parameters
        asses: bool 

    Returns:
        valids: number of fixed outputs
        df_output: dataframe containing output (either correct or incorrect) & original input
    """
    ## account for tokens that are not yet in SRC without changing existing SRC token embeddings
    errors = TabularDataset(
        path=error_source,
        format="csv",
        skip_header=False,
        fields={"SMILES": ("src", SRC)},
    )

    errors_loader = Iterator(
        errors,
        batch_size=64,
        device=device,
        sort=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
    )
    model.load_state_dict(torch.load(f=out + ".pkg"))
    # add option to use different iterator maybe?
    print("Correcting invalid SMILES")
    print(f"Number of invalid inputs: {len(errors.examples)}")
    valids, df_output = model.translate(errors_loader)
    #df_output.to_csv(f"{error_source}_fixed.csv", index=False)
    print(f"Finished, number of fixed outputs: {valids}")

    return valids, df_output
