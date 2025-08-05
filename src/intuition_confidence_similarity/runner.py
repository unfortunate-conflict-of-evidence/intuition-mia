import experimenter
import argparse
from datetime import date

def str_to_bool(v):
    '''
    Helper function to convert command-line strings to booleans.
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Run a single experiment.')

    # Dataset is required since it has no default value
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--percent', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='cnn')
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--same_class', type=str_to_bool, default=False)
    # Do not set a default here, so its value will be None if not provided
    parser.add_argument('--folder_name', type=str, required=False)

    args = parser.parse_args()

    # ---- Dynamic Default Folder Name ----
    # Check if the folder_name was provided on the command line
    if args.folder_name is None:
        today = date.today()
        month_number = today.month
        day_number = today.day
        
        if args.same_class:
            class_string = "same-class"
        else:
            class_string = "diff-class"

        args.folder_name = (
            f"{month_number}-{day_number}-{args.model_type}-{args.dataset}-"
            f"{class_string}-train-ratio-{args.train_ratio}"
        )
    # -------------------------------

    print(args)

    # Run experiment
    experimenter.raw_experiment_parallel(
        args.dataset,
        args.train_ratio,
        args.percent,
        args.model_type,
        args.num_trials,
        args.same_class,
        args.folder_name
    )
    experimenter.zip_folder(args.folder_name, args.folder_name)

if __name__ == '__main__':
    main()