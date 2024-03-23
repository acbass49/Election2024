# Election Model Using Bayesian Heirarchal Regression

I created an election model using hierarchal beta regression of the latest polls for 2024 US Presidential Election. You can follow daily updates on my [2024 US Presidential Election Dashboard](https://alexbass.me/projects/election_model_2024/) and I wrote up a few notes about this on my blog [here](https://alexbass.me/posts/election_2024/).

Daily updated predictions, election simulations, and tracking data will be stored in the data folder for use by others if desired.

![](data_pipeline.png)

### My data pipeline is visualized above and decribed below. This runs daily via github actions.

1. Load the polling data freely available on fivethirtyeight from a URL.
2. Process the data in Python
3. Estimate the model in Python (pymc)
4. Save the daily outputs in this repository
5. Later in the day, my dashboard loads these outputs (also via github actions) and generates three interactive figures displayed on my quarto website. This step is done in my quarto website repository; not in this one.

If you are looking for technical description of the model methodology, that is [here](https://alexbass.me/posts/election_2024/methodology.pdf).

Feel free to clone my work or use in any way.
