from schnetpack.datasets import QM9
import schnetpack.transform as trn


def get_dataloader():
    qm9data = QM9(
        './qm9.db',
        batch_size=1,
        num_train=110000,
        num_val=10000,
        transforms=[trn.ASENeighborList(cutoff=5.), trn.CastTo32()],
        property_units={QM9.gap: 'eV'},
        num_workers=1,
        pin_memory=True,  # set to false, when not using a GPU
        load_properties=[QM9.gap],  # only load U0 property
        distance_unit="Ang",
        remove_uncharacterized=True,
        pin_memory_device="cuda:0",
    )
    qm9data.prepare_data()
    qm9data.setup()
    return qm9data


if __name__ == '__main__':
    qm9data = get_dataloader()
    print(qm9data.dataset[0])
