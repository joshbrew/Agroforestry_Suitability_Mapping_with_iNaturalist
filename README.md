## Docs TBD, 

Look at Collection.md and each folder for how to start gathering.

Look at iNaturalistOccurrences and Suitability folders for how to start processing that data, lots of assembly required.

## Initial Results

Note our results are using an "artist's touch" to manually adjust eta-squared priors and manually control the blending between the ML and eta-squared results. The actual realism is species-dependent and requires more survey data, but this is already a great result running on fairly naive assumptions.

Eta-squared style empirically-weighted suitability scoring, with stress and reliability modifiers, compared to actual habitat ranges. We next added a machine learning habitat classifier to blend with this.
<img width="1780" height="1454" alt="image" src="https://github.com/user-attachments/assets/9f915fe0-b86c-4b35-84fe-7b45877b27c5" />
<img width="1371" height="730" alt="image" src="https://github.com/user-attachments/assets/fb658554-135e-4227-a258-1bc02b052ac4" />
<img width="1545" height="734" alt="image" src="https://github.com/user-attachments/assets/a567d6eb-4c36-4465-b8ea-25f9bc63e627" />
<img width="1617" height="1483" alt="image" src="https://github.com/user-attachments/assets/f5ffb234-2621-4ec4-8c5d-d616468f58f1" />
<img width="2010" height="696" alt="image" src="https://github.com/user-attachments/assets/82be1162-5cb6-4fb0-93cc-ecffe6ece130" />

XGBoost/ExtraTrees result
<img width="1635" height="1386" alt="image" src="https://github.com/user-attachments/assets/a58fbfad-7147-4e72-a31d-62e9f050030c" />

Blending eta-squared with the classifier probiabilites gives the most convincing results, e.g. the deserts truly are not suitable for most of the plants here:
<img width="1604" height="1419" alt="image" src="https://github.com/user-attachments/assets/a2f28052-5202-47d8-b2c2-9e6996bbac0a" />

Pseudotsuga menziesii (Douglas Fir) is the most generalized, Alnus Rubra (Red Alder) likes the hills, Arbutus menziesii (Pacific Madrone) favors the savannahs in the Willamette valley, and Kopsiopsis strobilaceae is parasitic to the Pacific Madrone, but found only in Southwestern Oregon and Northwestern California. Overall we get great overlap with their actual ranges and habitat preferences, with distinct hill and valley favoring species, and the parasitic species occupying a subsection of the wider Madrone range. 
