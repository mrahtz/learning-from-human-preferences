#!/usr/bin/env python3

import argparse
import pickle

from dot_utils import predict_preference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pref_db")
    args = parser.parse_args()

    with open(args.pref_db, 'rb') as pkl_file:
        pref_db = pickle.load(pkl_file)

    n_correct = 0
    n_wrong = 0

    updated_prefs = []

    for pref in pref_db:
        s1, s2, mu = pref
        predicted_mu = predict_preference(s1, s2)
        if mu == predicted_mu:
            n_correct += 1
        else:
            n_wrong += 1

        updated_prefs.append((s1, s2, predicted_mu))

    print(n_correct, n_wrong)

    with open('pref_db_fixed.pkl', 'wb') as pkl_file:
        pickle.dump(updated_prefs, pkl_file)
