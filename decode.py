"""
Decode event predictions from ACE-style to COVID-style.
"""

import json
import itertools
import copy


def decode_one(predicted_events_sent):
    new_predictions = []

    for pred in predicted_events_sent:
        # Break it down into trigger and argument categories.
        trig = pred[0]
        args = pred[1:]
        args0 = sorted([arg for arg in args if arg[2] == "ARG0"])
        args1 = sorted([arg for arg in args if arg[2] == "ARG1"])
        # If it doesn't have both types of arguments, skip it.
        if not (bool(args0) and bool(args1)):
            continue
        for arg0, arg1 in itertools.product(args0, args1):
            new_prediction = [trig, arg0, arg1]
            new_predictions.append(new_prediction)

    new_predictions = sorted(new_predictions, key=lambda entry: entry[0][0])
    return new_predictions


def decode(in_data):
    data = copy.deepcopy(in_data)  # We're going to mutate the data in-place.

    for doc in data:
        new_event_predictions = []
        for predicted_events_sent in doc["predicted_events"]:
            decoded_events = decode_one(predicted_events_sent)
            new_event_predictions.append(decoded_events)

        assert len(new_event_predictions) == len(doc["predicted_events"])
        doc["_old_predicted_events"] = doc["predicted_events"]
        doc["predicted_events"] = new_event_predictions

    return data
