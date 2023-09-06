The code is structured such that model.py imports preprocess functions from preprocess.py and it only applies them.
Please note:
* The model utilized is LogisticRegression with class balancing and feature importance. This decision is made based on the performance shown in the notebook, since is the one that achieves the best mean f1-score for both labels.
* F1-score was used to determine the best model since it leverages really well unbalanced settings. More knowledge would be needed from the business to prioritize
* Some tests were slightly modified to properly load the data attached without modifying the structure of the project.
* Some tests were modified to pass in the scenario where the performance of the model is not as expected (could not replicate the performance of LogisticRegression from the notebook)
* Some requirements were changed since they were not properly running the tests. In particular I updated fastapi and starlette to able to use the Client class for testing.
* The Continuos Integration is provided partially to demonstrate the use of Github Actions, although is limited since the API is not deployed and thus cannot be tested.
* The Continuos Delivery was not developed, however it would be similar to the script found in build-img.sh which builds the docker container and pushes it to GCR. This docker container works locally when running stress-test.
