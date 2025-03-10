% Define the subject
subj = 4; % Subject 4

% Define the trials
trials = ["walk_09", "walk_18", "walk_27", "walk_36", "walk_45", "walk_54",...
          "run_63", "run_81", "run_99", "jump", "squat", "land", "lunge"];

% Initialize a structure to hold the data
allTrialsData = struct();

% Loop through the trials
for tr_id = 1:length(trials)
    trial = trials(tr_id);
    
    % Construct the file name
    if subj < 10
        file_name = strcat('Subj0', num2str(subj),...
            '/Subj0', num2str(subj), '_', trial, '.mat');
    else
        file_name = strcat('Subj', num2str(subj),...
            '/Subj', num2str(subj), '_', trial, '.mat');
    end
    
    % Load the .mat file
    try
        load(file_name);
    catch
        fprintf('Could not load file: %s. Skipping this trial.\n', file_name);
        continue; % Skip to the next trial
    end
    
    % Extract joint angle data
    try
        joint_angles = Datastr.Resample.Sych.IKAngData;
        joint_angles_labels = Datastr.Resample.Sych.IKAngDataLabel;
    catch
        fprintf('Could not extract joint angle data from %s. Skipping this trial.\n', file_name);
        continue; % Skip to the next trial
    end
    
    % Store the joint angles and labels in the structure
    allTrialsData.(trial).joint_angles = joint_angles;
    allTrialsData.(trial).joint_angles_labels = joint_angles_labels;
    
    fprintf('Successfully extracted and stored joint angles for Subject %d, trial %s.\n', subj, trial);
end

% Save the extracted data to a new .mat file
output_file_name = sprintf('Subj0%d_all_trials_joint_angles.mat', subj);
save(output_file_name, 'allTrialsData');

fprintf('Extracted joint angle data for all trials of Subject %d saved to %s.\n', subj, output_file_name);
