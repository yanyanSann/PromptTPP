import random
from neural_tpp.preprocess.torch_dataset import TorchTPPDataset, create_torch_dataloader


def make_raw_data():
    data = [
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 0}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
    ]
    for i, j in enumerate([2, 5, 3]):
        start_time = 0
        for k in range(j):
            delta_t = random.random()
            start_time += delta_t
            data[i].append({"time_since_last_event": delta_t,
                            "time_since_start": start_time,
                            "type_event": random.randint(0, 10)
                            })
    data[1][3]["time_since_start"] = data[1][2]["time_since_start"]
    data[1][3]["time_since_last_event"] = data[1][2]["time_since_last_event"]
    data[1][4]["time_since_start"] = data[1][2]["time_since_start"]
    data[1][4]["time_since_last_event"] = data[1][2]["time_since_last_event"]
    data[1][5]["time_since_start"] = data[1][2]["time_since_start"]
    data[1][5]["time_since_last_event"] = data[1][2]["time_since_last_event"]

    return data


def main():
    data = make_raw_data()
    data_config = dict(event_num=11,
                       source_data=data,
                       pad_end=False)
    dataset = TorchTPPDataset(data_config)
    dataloader = create_torch_dataloader(dataset, batch_size=2)
    for i, _batch in enumerate(dataloader):
        a, b, c, d, e, f = _batch
        print(a.shape, b.shape, c.shape)
        print("event_type")
        print(c)
        print("time_seq")
        print(a)
        print("time_delta_seq")
        print(b)
        print("batch_non_pad_mask")
        print(d)
        print("attention_mask")
        print(e)
        print("type_mask")
        print(f)


if __name__ == '__main__':
    main()
