# Preparing the Dataset with Builder Script or TFDS Module

1. For this example, we will be using the [**10 Monkey Species**](https://www.kaggle.com/datasets/slothkong/10-monkey-species) dataset.
    
    The directory structure of this dataset is as follows:
    ```
    monkey_species_dataset
    |-- __init__.py
    |-- monkey_labels.txt
    |-- training
    |   `-- training
    |       |-- n0
    |       |-- n1
    |       |-- n2
    |       |-- n3
    |       |-- n4
    |       |-- n5
    |       |-- n6
    |       |-- n7
    |       |-- n8
    |       `-- n9
    |           |-- n9151jpg
    |           `-- n9160.png
    `-- validation
        `-- validation
            |-- n0
            |-- n1
            |-- n2
            |-- n3
            |-- n4
            |-- n5
            |-- n6
            |-- n7
            |-- n8
            `-- n9
    ```

2. Next let us setup [`wandb-addons`](https://github.com/soumik12345/wandb-addons). We can do this using the following command:
    ```shell
    git clone https://github.com/soumik12345/wandb-addons
    pip install -q wandb-addons[dataset]
    ```

    This would install `wandb-addons` and also optional dependencies including [`tensorflow`](https://www.tensorflow.org/) and `tfds-nightly`, the nightly release of [`tensorflow-datasets`](https://www.tensorflow.org/datasets).

3. Now, let us `cd` into the `monkey_species_dataset` directory and initialize the tensorflow datasets template files, which would be used for interpreting and registering features from our dataset.
    ```shell
    cd monkey_species_dataset
    # Create `monkey_species_dataset/monkey_species` template files.
    tfds new monkey_species 
    ```

    This would create a directory with the following structure inside the directory `monkey_species_dataset`:

    ```
    monkey_species
       |-- CITATIONS.bib
       |-- README.md
       |-- TAGS.txt
       |-- __init__.py
       |-- checksums.tsv
       |-- dummy_data
       |   `-- TODO-add_fake_data_in_this_directory.txt
       |-- monkey_species_dataset_builder.py
       `-- monkey_species_dataset_builder_test.py
    ```

    The complete directory structure of `monkey_species_dataset` at this point is going to something like:

    ```
    monkey_species_dataset
    |-- __init__.py
    |-- monkey_labels.txt
    |-- monkey_species
    |   |-- CITATIONS.bib
    |   |-- README.md
    |   |-- TAGS.txt
    |   |-- __init__.py
    |   |-- checksums.tsv
    |   |-- dummy_data
    |   |   `-- TODO-add_fake_data_in_this_directory.txt
    |   |-- monkey_species_dataset_builder.py
    |   `-- monkey_species_dataset_builder_test.py
    |-- training
    |   `-- training
    |       |-- n0
    |       |-- n1
    |       |-- n2
    |       |-- n3
    |       |-- n4
    |       |-- n5
    |       |-- n6
    |       |-- n7
    |       |-- n8
    |       `-- n9
    |           |-- n9151jpg
    |           `-- n9160.png
    `-- validation
        `-- validation
            |-- n0
            |-- n1
            |-- n2
            |-- n3
            |-- n4
            |-- n5
            |-- n6
            |-- n7
            |-- n8
            `-- n9
    ```

    !!! note "Note"
        The name with which you initialize the `tfds new` command would be used as the `name` of your dataset.

4. Now we will write our **dataset builder** in the file `monkey_species_dataset/monkey_species/monkey_species_dataset_builder.py`. This logic for writing a dataset builder is exactly similar to that of creating the same for HuggingFace Datasets or a vanilla TensorFlow dataset.

    !!! note "Note"
        Alternative to step 3, you could also simply inclide a builder script `<dataset_name>.py` into the `monkey_species_dataset` directory, instead of creating the TFDS module.

    !!! example "You can refer to the following examples"
        - [An example of a dataset with a custom builder script](https://wandb.ai/geekyrakshit/monkey-dataset/artifacts/dataset/monkey_species/v1/files)
        - [An example of a dataset with a TFDS module](https://wandb.ai/geekyrakshit/monkey-dataset/artifacts/dataset/monkey_species/v2/files)
    
    !!! info "You can refer to the following guides for writing builder scripts"
        - [Writing custom datasets using `tfds`](https://www.tensorflow.org/datasets/add_dataset).
        - [HuggingFace: Create a dataset](https://huggingface.co/docs/datasets/main/en/create_dataset).
        - [HuggingFace: Create an Image Dataset](https://huggingface.co/docs/datasets/main/en/image_dataset).

5. Now that our dataset is ready with the specifications for loading the features, we can upload it to our Weights & Biases project as an [artifact](https://docs.wandb.ai/guides/artifacts) using the `upload_dataset` function, which would verify if the dataset build is successful or not before uploading the dataset.

    ```python
    import wandb
    from wandb_addons.dataset import upload_dataset

    # Initialize a W&B Run
    wandb.init(project="my-awesome-project", job_type="upload_dataset") 
    
    # Note that we should set our dataset name as the name of the artifact
    upload_dataset(name="my_awesome_dataset", path="./my/dataset/path", type="dataset")
    ```

    In order to load this dataset in your ML workflow you can simply use the `load_dataset` function:

    ```python
    from wandb_addons.dataset import load_dataset

    datasets, dataset_builder_info = load_dataset("entity-name/my-awesome-project/my_awesome_dataset:v0")
    ```

    !!! note "Note"
        - In the `upload_dataset` function by default convert a registered dataset to TFRecords (like [this](https://wandb.ai/geekyrakshit/monkey-dataset/artifacts/dataset/monkey_species/tfrecords/overview) artifact). You can alternatively upload the dataset in its original state along with the added TFDS module containing the builder script by simply setting `upload_tfrecords` parameter to `False`.
        - Note that this won't affect loading the dataset using `load_dataset`, dataset loading from artifacts would work as long as the artifact contains either the TFRecords or the original dataset with the TFDS module.
        - The TFRecord artifact has to follow the specification specified in [this](https://www.tensorflow.org/datasets/external_tfrecord) guide. However, if you're using the `upload_dataset` function, you don't need to worry about this.
    
    You can take a look at [**this**](https://wandb.ai/geekyrakshit/monkey-dataset/artifacts/dataset/monkey_species/v1/files) artifact that demonstrates the aforementioned directory structure and builder logic.
