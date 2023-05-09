# datathon23
Code from the ACE Datathon 2023 in Zürich.
Team BONK, challenge provided by D-ONE.

## Tasks
We will evaluate you on your success at three tasks. We have numbered them as 1, 2 and 3 below; this is for clarity, and we suggest tackling them in parallel.
### Task 1: Preprocessing
You will be working on real world, only lightly curated data. This data will require some purging and transforming; this is your first task. You will have to explain what pre-processing you did and why.
In the app, each training is tagged as one of 5 types. These are proposed by the app, and thus not always reliable – the user may or may not have followed the recommendation. The types are:
● LONG JOG: A long jog is a type of endurance training where the runner maintains a moderate pace for an extended distance or duration.
● INTERVAL: Interval training involves alternating periods of high-intensity exercise with periods of lower-intensity recovery or rest.
● STEADY JOG: A steady jog is a type of run that maintains a consistent, comfortable, medium pace throughout the entire session.
● LOW INTENSITY: Low-intensity runs/walks are slower-paced workouts, focusing on maintaining a comfortable and relaxed pace.
● RACE: competitive running event.
### Task 2: Predicting the type of training
Your second task is to create an algorithm that, given the log and metadata of a run, predicts the type of training among the five above.
Out of the ~16k trainings, 250 have column type empty. You will have to provide your algorithm’s prediction of the missing value for the 250. For convenience, we provide the table exam_dataset, containing these 250 pairs of user_id and training_id isolated, and a blank column for you to fill in your predicted category.
### Task 3: Developing a personalized metric
Finally, the users of the app are interested in personalized metrics that track their fitness over time. Your third task is to design and prototype such a metric. This should be a time-dependent value, which allows users to compare their current to past performance.
Other than this, you have complete creative control. You are free to enrich the data with any other publicly available data source. We will evaluate your algorithm on the grounds of originality, intuitiveness and sophistication.