from mne_bids import BIDSPath, read_raw_bids
import mne

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

from temporal_norm.config import DATA_PATH

mne.set_log_level("warning")


def extract_epochs(raw, eog=False, emg=False, chunk_duration=30.0):
    """Extract non-overlapping epochs from raw data.
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to be windowed.
    chunk_duration : float
        Length of a window.
    Returns
    -------
    np.ndarray
        Epoched data, of shape (n_epochs, n_channels, n_times).
    np.ndarray
        Event identifiers for each epoch, shape (n_epochs,).
    """
    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=chunk_duration
    )

    # create a new event_id that unifies stages 3 and 4
    # event_id = {}
    # if 1 in events[:, 2]:
    #     event_id["Sleep stage W"] = 1
    # if 2 in events[:, 2]:
    #     event_id["Sleep stage 1"] = 2
    # if 3 in events[:, 2]:
    #     event_id["Sleep stage 2"] = 3
    # if 4 in events[:, 2]:
    #     event_id["Sleep stage 3/4"] = 4
    # if 5 in events[:, 2]:
    #     event_id["Sleep stage R"] = 5

    classes = np.unique(events[:, 2])

    event_id = {}
    event_name = ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage R"]
    for c in classes:
        event_id[event_name[c-1]] = c

    tmax = 30.0 - 1.0 / raw.info["sfreq"]  # tmax in included
    picks = mne.pick_types(raw.info, eeg=True, eog=eog, emg=emg)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        picks=picks,
        preload=True,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )

    return epochs.get_data(), epochs.events[:, 2] - 1


def apply_scaler(data):
    data -= np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)
    std[std == 0] = 1
    data /= std
    return data


def read_raw_bids_with_preprocessing(
    bids_path, scaler, eog=False, emg=False, to_microvolt=True
):
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    data, event = extract_epochs(raw, eog, emg)
    if to_microvolt:
        data *= 1e6
    if scaler:
        data = apply_scaler(data)

    return data.astype("float32"), event.astype("int64")


def load_data(
    n_subjects,
    data_path,
    eog=False,
    emg=False,
    scaler="sample",
):
    """XXX docstring"""
    all_data = list()
    all_events = list()
    subject_ids = list()
    sessions = list()
    datatype = "eeg"
    suffix = "eeg"
    all_sub = (
        pd.read_csv(
            data_path / "participants.tsv",
            delimiter="\t",
        )["participant_id"]
        .transform(lambda x: x[4:])
        .tolist()
    )
    if n_subjects == -1:
        n_subjects = len(all_sub)
    pbar = tqdm(total=n_subjects)

    for subj_id in range(n_subjects):
        subject = all_sub[subj_id]
        try:

            bids_path = BIDSPath(
                datatype=datatype,
                root=data_path,
                suffix=suffix,
                task="sleep",
                subject=subject,
            )
            list_sessions = []
            for path_ in bids_path.match():
                ses = path_.session
                if ses not in list_sessions:
                    list_sessions.append(ses)

            for ses in list_sessions:
                bids_path = BIDSPath(
                    datatype=datatype,
                    root=data_path,
                    suffix=suffix,
                    task="sleep",
                    session=ses,
                    subject=subject,
                )
                data, events = read_raw_bids_with_preprocessing(
                    bids_path, scaler, eog, emg
                )
                all_data.append(data)
                all_events.append(events)
                subject_ids.append(subj_id)
                sessions.append(ses)

        except ValueError:
            print("That was no valid epoch.")

        except PermissionError:
            print("subject no valid")

        except TypeError:
            print("subject no valid")

        except FileNotFoundError:
            print("File not found")

        pbar.update(1)

    # Concatenate into a single dataset
    pbar.close()
    print("number of subjects:", len(subject_ids))
    return all_data, all_events, subject_ids, sessions


def load_dataset(
    n_subjects,
    dataset_name,
    eog=False,
    emg=False,
    data_path=None,
    scaler=True,
):
    if dataset_name == "MASS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "MASS" / "SS3" / "4channels-eeg_eog_emg"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "ABC":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "ABC" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "CHAT":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "CHAT" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler, )

    if dataset_name == "CFS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "CFS" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "HOMEPAP":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "HOMEPAP" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "CCSHS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "CCSHS" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "SOF":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "SOF" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "MROS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "MROS" / "2channels"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "Physionet":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "Physionet" / "4channels-eeg_eog_emg"
        return load_data(n_subjects, data_path, eog, emg, scaler,)

    if dataset_name == "SHHS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "SHHS" / "4channels-eeg_eog_emg"
        return load_data(n_subjects, data_path, eog, emg, scaler,)
