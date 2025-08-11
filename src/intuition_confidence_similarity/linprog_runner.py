import linprog_experimenter
import argparse
from datetime import date
from runner import str_to_bool
from experimenter import zip_folder

def main():
    parser = argparse.ArgumentParser(description='Run a linear programming experiment.')

    parser.add_argument('--train_size', type=int, default=10,
                        help='Number of training data points.')
    parser.add_argument('--grid_size', type=float, default=1000.0,
                        help='Size of the data grid.')
    parser.add_argument('--granularity', type=float, default=1.0,
                        help='Granularity of the data points on the grid.')
    parser.add_argument('--percent', type=int, default=5,
                        help='The top/bottom percentage to analyze.')
    parser.add_argument('--num_trials', type=int, default=100,
                        help='Number of trials to run for each experiment.')
    parser.add_argument('--same_class', type=str_to_bool, default=False,
                        help='Whether to find the nearest neighbor from the same class.')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='The name of the experiment folder.')
    parser.add_argument('--ideal_slope', type=float, default=1.0,
                        help='The slope of the ideal separating line.')
    parser.add_argument('--ideal_intercept', type=float, default=0.0,
                        help='The intercept of the ideal separating line.')
    parser.add_argument('--base_folder', type=str, required=False,
                        help='The name of the base directory.')

    args = parser.parse_args()

    # ---- Dynamic Default Folder Name ----
    # Check if the folder_name was provided on the command line
    if args.base_folder is None:
        today = date.today()
        month_number = today.month
        day_number = today.day
        
        if args.same_class:
            class_string = "same-class"
        else:
            class_string = "diff-class"

        args.base_folder = (
            f"{month_number}-{day_number}-linprog-"
            f"{class_string}-train-size-{args.train_size}"
        )
    # -------------------------------

    print(args)

    # Run experiment
    linprog_experimenter.execute_linprog_experiments(
        args.exp_name, 
        args.ideal_slope, 
        args.ideal_intercept, 
        args.train_size, 
        args.grid_size, 
        args.granularity, 
        args.num_trials, 
        args.percent, 
        args.same_class, 
        args.base_folder
    )

if __name__ == '__main__':
    main()