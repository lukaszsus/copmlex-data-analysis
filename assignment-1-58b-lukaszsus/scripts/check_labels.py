from utils.data_loader import load_mails, load_labels


def check_labels():
    df_labels = load_labels()
    higher_level = list(df_labels[df_labels["Label"] == 2]['ID'])
    print(f"Higher level management: {higher_level}")
    print(f"Length: {len(higher_level)}")

    mid_level = list(df_labels[df_labels["Label"] == 1]['ID'])
    print(f"Mid level management: {mid_level}")
    print(f"Length: {len(mid_level)}")


if __name__ == '__main__':
    check_labels()
