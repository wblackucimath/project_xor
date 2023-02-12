import logging
from proj_xor.models import ProjXORWrapper
from proj_xor.plots import plot_data

def main(logging_level=logging.WARNING):
    logging.basicConfig(level=logging_level)
    logging.info("Entering main method.")

    M = ProjXORWrapper()

    M.fit(
        save_plots=True,
        show_plots=False,
        save_dfs=True,
        show_dfs=False,
        monitor_freq=5,
    )

    plot_data.plot_model_performance(M.get_model(), show_plt=False)

    logging.info("Exiting main method.")


if __name__ == "__main__":
    main(logging_level=logging.INFO)
