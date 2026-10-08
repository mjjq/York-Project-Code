from argparse import ArgumentParser

from jorek_tools.island_width.calibrated_island_width import avg_island_width_to_outboard
from chease_tools.dr_term_at_q import read_columns
from experiments.ntm_modelling.mre_time_series import MeasuredIslandWidth

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'chease_cols',
        type=str,
        help='Path to chease_cols.out file'
    )
    parser.add_argument(
        '-w', '--w-average',
        type=float,
        help="Helically averaged island width"
    )
    parser.add_argument(
        '-werr', '--w-average-error',
        type=float,
        help="Helically averaged island width error",
        default=0.0
    )

    parser.add_argument(                                  
        '-m', '--poloidal-mode',                          
        help="List of poloidal mode numbers to evaluate", 
        default=2,                                        
        type=int                                          
    )                                                     
    parser.add_argument(                                  
        '-n', '--toroidal-mode', type=int,                
        help='Toroidal mode number',                      
        default=1,                                      
    )

    args = parser.parse_args()

    cols = read_columns(args.chease_cols)

    w_measured = MeasuredIslandWidth(
        0.0,
        args.w_average,
        args.w_average_error,
        True
    )

    w_outboard = avg_island_width_to_outboard(
        cols,
        w_measured,
        args.poloidal_mode,
        args.toroidal_mode
    )

    print(w_outboard)

