# iNaturalist + global GIS species suitability mapping with eta-squared and XGBoost/ExtraTrees

See preliminary results below.

This software lets you make individual and combined suitability maps for any species captured in the iNaturalist dataset, by pulling a rich set of climate, geological, and topological data from other freely available datasets. This lets you map possible habitat ranges for plants, the idea being you may want to develop land with many kinds of species, e.g. for permaculture or industrial agroforestry. 

<img width="1480" height="396" alt="ChatGPT Image Mar 20, 2026, 01_46_45 AM" src="https://github.com/user-attachments/assets/60e4acde-96b5-4607-9c38-07bd36599930" />

This project requires ~1 TB of data to run yourself:
- bulk global iNaturalist species occurrences via GBIF.org
- Terraclimate 12 month global climate summaries (latest or multi-year)
- SoilGrids 2.0 global grids
- Global TWI 
- Global DEM via HydroSHEDs + derived tiles (scripts supplied)
- Global GLiM lithographs.
- MCD12Q1 land use classification map (latest)

This system is memory-optimized to run on laptops, I did it with a USB drive to make it extra slow for myself and focus on optimization, so much of the data requires transforming into better COG TIF format or indexing for quick CSV lookup. All scripts are provided, including an enrichment script that lets you define any taxa levels from iNaturalist and enrich cumulative csvs to run through our suitability mapping programs. This all needs documentation or you can feed files into a good LLM to get the workflow spelled out for you. 

## Docs TBD, 

Look at Collection.md and each folder for how to start gathering.

Look at iNaturalistOccurrences and Suitability folders for how to start processing that data, lots of assembly required.

## Initial Results

Note our results are using an "artist's touch" to manually adjust eta-squared priors and manually control the blending between the ML and eta-squared results. The actual realism is species-dependent and requires more survey data, but this is already a great result running on fairly naive assumptions.

Blending eta-squared with the classifier probabilites gives the most convincing results, e.g. the deserts truly are not suitable for most of the plants here:
<img width="1137" height="1012" alt="image" src="https://github.com/user-attachments/assets/6816995d-2600-44ff-be60-2e00ef9c9e87" />


Eta-squared style empirically-weighted suitability scoring, with stress and reliability modifiers, compared to actual habitat ranges. We next added a machine learning habitat classifier to blend with this.
<img width="1212" height="1025" alt="image" src="https://github.com/user-attachments/assets/2a9eddec-1c83-4bff-b71b-677252ba565f" />
<img width="1181" height="694" alt="image" src="https://github.com/user-attachments/assets/0b814f00-8ee7-436e-a09c-4569dcf2a61a" />
<img width="1782" height="847" alt="image" src="https://github.com/user-attachments/assets/54db69d9-c9dc-46a9-8ff8-dca0417cbcec" />
<img width="1163" height="1054" alt="image" src="https://github.com/user-attachments/assets/9561495c-bc84-4e52-8c08-c7f936a84bf3" />

Initial results overlap well with known habitat, using well known Oregon species as our test case. Future results will show "agroforestry" profiles where we have valid overlap for dozens of useful cultivatable species, but we're still solving some memory performance problems for scaling to millions of points with these huge datasets as we're using a slow device. If you have a nice workstation with a very fast SSD you should not have a huge issue getting quicker results.


XGBoost/ExtraTrees result
<img width="1172" height="975" alt="image" src="https://github.com/user-attachments/assets/bc5d7070-f24d-4e75-904d-b0977ef64162" />

Model performance:
<img width="2045" height="705" alt="image" src="https://github.com/user-attachments/assets/a7f5a682-7fae-4b4b-b336-9fd084e05797" /> The low F1 scores here are more due to the extrapolating rather than the original classification accuracy, as we deliberaly are weakening it to get a larger suitability area.

Leaky XGBoost model that overtunes around the actual observation sites (minus coordinates), useful for blending better from occurrence data ground truth:
<img width="1427" height="1199" alt="image" src="https://github.com/user-attachments/assets/58a74726-afba-4077-99be-d16c0081e4fa" /> This was our first attempt but it didn't do background sampling correctly, however it still has some usefulness as a narrower model.

Leaky model (overtunes around observation sites), very high F1 due to less extrapolation:
<img width="2010" height="696" alt="image" src="https://github.com/user-attachments/assets/82be1162-5cb6-4fb0-93cc-ecffe6ece130" />

We also created a community model on top of this to look at multi-species probabilities, we are still testing it.


Numeric comparisons:
<img width="1757" height="1635" alt="image" src="https://github.com/user-attachments/assets/c8688028-c6a7-4213-9ff6-aefee6378fca" />
<img width="2646" height="1470" alt="image" src="https://github.com/user-attachments/assets/0d15095e-27e5-4709-9fc1-d19f06ab3fad" />
Raw occurrence data, you'll see that the ML model has very strong overlap with the clusters here:
<img width="1612" height="1349" alt="image" src="https://github.com/user-attachments/assets/618dbe74-54c2-420a-b40f-7e02f57a19b9" />



Pseudotsuga menziesii (Douglas Fir) is the most generalized, Alnus Rubra (Red Alder) likes the hills, Arbutus menziesii (Pacific Madrone) favors the savannahs in the Willamette valley, and Kopsiopsis strobilaceae is parasitic to the Pacific Madrone, but found only in Southwestern Oregon and Northwestern California. Overall we get great overlap with their actual ranges and habitat preferences, with distinct hill and valley favoring species, and the parasitic species occupying a subsection of the wider Madrone range. 

### Contribute

This is free to use for any reason under an MIT License. It's not trivial to set up by any means but we may streamline it more as we continue exploring this framework. If you find any interest in using or improving it, feel free to fork or contribute to this repository. We're all in this together!
