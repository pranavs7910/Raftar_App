% 1. Load the raw ADC data from your text file
raw_data = load('apps1_app.txt');

% 2. Define the Time Vector
% Assuming a 10ms (0.01s) sampling rate from your rover's log
fs = 100; % 100 Hz
t = (0:length(raw_data)-1)' / fs;

% 3. Create a Timeseries object
% This is the format Simulink requires to 'see' the data over time
apps1_signal = timeseries(raw_data, t);

% 4. (Optional) Repeat for APPS2 if you have that file
% apps2_data = load('apps2_app.txt');
% apps2_signal = timeseries(apps2_data, t);

fprintf('Data imported. variable "apps1_signal" is ready for Simulink.\n');

% 1. Load the raw ADC data from your text file
raw_data = load('apps2_app.txt');

% 2. Define the Time Vector
% Assuming a 10ms (0.01s) sampling rate from your rover's log
fs = 100; % 100 Hz
t = (0:length(raw_data)-1)' / fs;

% 3. Create a Timeseries object
% This is the format Simulink requires to 'see' the data over time
apps2_signal = timeseries(raw_data, t);

% 4. (Optional) Repeat for APPS2 if you have that file
% apps2_data = load('apps2_app.txt');
% apps2_signal = timeseries(apps2_data, t);

fprintf('Data imported. variable "apps2_signal" is ready for Simulink.\n');
% =========================================================================
% APPS Signal Processing & Motor Torque Calculation
% Accelerator Pedal Position Sensor (APPS) Dual-Channel Safety System
% =========================================================================
% Description:
%   Processes two APPS signals from a 12-bit ADC, normalizes them to 0-100%
%   pedal travel, applies median filtering, performs a plausibility/safety
%   check, and maps the result to a quadratic motor torque output.
%
% Signal Ranges:
%   APPS1: 0V  - 1.68V  (ADC: 0    - ~2084)
%   APPS2: 1.7V - 3.3V  (ADC: ~2109 - 4095)
% =========================================================================

clc; clear; close all;

%% -------------------------------------------------------------------------
%  SECTION 1: CONFIGURATION & CONSTANTS
% -------------------------------------------------------------------------

% --- ADC Parameters ---
ADC_RESOLUTION  = 4095;     % 12-bit ADC max count
V_REF           = 3.3;      % Reference voltage (V)
V_PER_COUNT     = V_REF / ADC_RESOLUTION;

% --- APPS1 Calibration (0V to 1.68V → 0% to 100%) ---
APPS1_ADC_MIN   = 0;                            % 0.00V
APPS1_ADC_MAX   = round(1.68 / V_PER_COUNT);    % ≈ 2084 counts

% --- APPS2 Calibration (1.7V to 3.3V → 0% to 100%) ---
APPS2_ADC_MIN   = round(1.70 / V_PER_COUNT);    % ≈ 2109 counts
APPS2_ADC_MAX   = ADC_RESOLUTION;               % 4095 counts

% --- Torque Map Parameters ---
T_MAX           = 230;      % Maximum torque (Nm)
P_DEAD          = 5.0;      % Deadband threshold (%)
N_EXPONENT      = 2;        % Quadratic exponent

% --- Safety Parameters ---
PLAUSIBILITY_THRESHOLD = 10.0;  % Max allowable difference between APPS (%)

% --- Filter Parameters ---
MEDIAN_FILTER_WIN = 5;      % Median filter window size (samples)

fprintf('=== APPS Torque Processor Initialized ===\n');
fprintf('ADC Resolution  : 12-bit (%d counts, %.4f V/count)\n', ADC_RESOLUTION+1, V_PER_COUNT);
fprintf('APPS1 Range     : ADC %d – %d  (%.2fV – 1.68V)\n', APPS1_ADC_MIN, APPS1_ADC_MAX, APPS1_ADC_MIN*V_PER_COUNT);
fprintf('APPS2 Range     : ADC %d – %d  (1.70V – %.2fV)\n', APPS2_ADC_MIN, APPS2_ADC_MAX, APPS2_ADC_MAX*V_PER_COUNT);
fprintf('Torque Map      : T = %.0f × (P - %.0f%%)^%d  [Nm]\n\n', T_MAX, P_DEAD, N_EXPONENT);

%% -------------------------------------------------------------------------
%  SECTION 2: DATA INPUT
% -------------------------------------------------------------------------

apps1_file = 'apps1_app.txt';
apps2_file = 'apps2_app.txt';

% --- Load APPS1 ---
if isfile(apps1_file)
    raw_apps1 = load(apps1_file);
    fprintf('[OK] Loaded APPS1: %d samples from "%s"\n', length(raw_apps1), apps1_file);
else
    warning('File "%s" not found. Generating synthetic APPS1 data for demonstration.', apps1_file);
    t_sim = linspace(0, 10, 2000)';
    % Simulated pedal: ramp up, hold, ramp down, with noise
    pedal_profile = [linspace(0, 1, 500)'; ones(500,1); linspace(1, 0.3, 400)'; ...
                     linspace(0.3, 0.8, 300)'; linspace(0.8, 0, 300)'];
    raw_apps1 = round(APPS1_ADC_MIN + pedal_profile .* (APPS1_ADC_MAX - APPS1_ADC_MIN) + ...
                      randn(2000,1) * 8);
    raw_apps1 = max(0, min(ADC_RESOLUTION, raw_apps1));
end

% --- Load APPS2 ---
if isfile(apps2_file)
    raw_apps2 = load(apps2_file);
    fprintf('[OK] Loaded APPS2: %d samples from "%s"\n', length(raw_apps2), apps2_file);
else
    warning('File "%s" not found. Generating synthetic APPS2 data for demonstration.', apps2_file);
    pedal_profile2 = pedal_profile;
    % Inject a plausibility fault between samples 800–850
    pedal_profile2(800:850) = pedal_profile2(800:850) + 0.18;
    raw_apps2 = round(APPS2_ADC_MIN + pedal_profile2 .* (APPS2_ADC_MAX - APPS2_ADC_MIN) + ...
                      randn(2000,1) * 8);
    raw_apps2 = max(0, min(ADC_RESOLUTION, raw_apps2));
end

% --- Align signal lengths ---
n_samples = min(length(raw_apps1), length(raw_apps2));
raw_apps1 = raw_apps1(1:n_samples);
raw_apps2 = raw_apps2(1:n_samples);
sample_idx = (1:n_samples)';

fprintf('Processing %d aligned samples.\n\n', n_samples);

%% -------------------------------------------------------------------------
%  SECTION 3: NORMALIZATION (ADC counts → 0–100% pedal travel)
% -------------------------------------------------------------------------

% Clamp raw ADC values to valid calibration ranges before scaling
clamped_apps1 = max(APPS1_ADC_MIN, min(APPS1_ADC_MAX, raw_apps1));
clamped_apps2 = max(APPS2_ADC_MIN, min(APPS2_ADC_MAX, raw_apps2));

% Linear normalization to percentage
norm_apps1 = ((clamped_apps1 - APPS1_ADC_MIN) / (APPS1_ADC_MAX - APPS1_ADC_MIN)) * 100;
norm_apps2 = ((clamped_apps2 - APPS2_ADC_MIN) / (APPS2_ADC_MAX - APPS2_ADC_MIN)) * 100;

%% -------------------------------------------------------------------------
%  SECTION 4: SIGNAL PROCESSING — Median Filter
% -------------------------------------------------------------------------

filt_apps1 = manual_medfilt(norm_apps1, MEDIAN_FILTER_WIN);
filt_apps2 = manual_medfilt(norm_apps2, MEDIAN_FILTER_WIN);

% Absolute difference between filtered signals
abs_diff = abs(filt_apps1 - filt_apps2);

fprintf('Signal Statistics (after filtering):\n');
fprintf('  APPS1 — Min: %.2f%%  Max: %.2f%%  Mean: %.2f%%\n', min(filt_apps1), max(filt_apps1), mean(filt_apps1));
fprintf('  APPS2 — Min: %.2f%%  Max: %.2f%%  Mean: %.2f%%\n', min(filt_apps2), max(filt_apps2), mean(filt_apps2));
fprintf('  |Diff| — Max: %.2f%%  Mean: %.2f%%\n\n', max(abs_diff), mean(abs_diff));

%% -------------------------------------------------------------------------
%  SECTION 5: SAFETY CHECK & TORQUE MAPPING
% -------------------------------------------------------------------------

% Average pedal travel from both sensors
pedal_avg = (filt_apps1 + filt_apps2) / 2;

% --- Plausibility Check ---
% FMEA Rule: If |APPS1 - APPS2| > 10%, torque = 0 (sensor fault detected)
plausibility_ok = abs_diff <= PLAUSIBILITY_THRESHOLD;   % logical array

% --- Torque Map: Quadratic with deadband ---
% If pedal travel <= deadband → torque = 0 (no creep)
% If pedal travel > deadband  → torque = T_max × ((P - P_dead) / (100 - P_dead))^n
% Note: travel is normalised so 100% input → 100% torque at T_max

torque_raw = zeros(n_samples, 1);
active = pedal_avg > P_DEAD;

% Normalise travel above deadband to 0–1 range before applying exponent
travel_norm = (pedal_avg(active) - P_DEAD) / (100 - P_DEAD);
torque_raw(active) = T_MAX .* (travel_norm) .^ N_EXPONENT;

% Apply plausibility gate — zero torque on fault
torque_output = torque_raw .* plausibility_ok;

% --- Fault Statistics ---
n_faults = sum(~plausibility_ok);
fault_pct = (n_faults / n_samples) * 100;
fprintf('Plausibility Faults : %d / %d samples (%.1f%%)\n', n_faults, n_samples, fault_pct);
fprintf('Peak Torque Output  : %.2f Nm\n', max(torque_output));
fprintf('Mean Torque Output  : %.2f Nm\n\n', mean(torque_output));

%% -------------------------------------------------------------------------
%  SECTION 6: PLOTTING  (downsampled for speed — all calcs used full data)
% -------------------------------------------------------------------------

% Downsample to ~5000 points just for plotting
PLOT_EVERY = max(1, floor(n_samples / 5000));
idx_ds     = 1:PLOT_EVERY:n_samples;
fprintf('Plotting %d points (1 in every %d) for speed...\n', length(idx_ds), PLOT_EVERY);

% Downsampled vectors
s_ds   = sample_idx(idx_ds);
a1_ds  = filt_apps1(idx_ds);
a2_ds  = filt_apps2(idx_ds);
d_ds   = abs_diff(idx_ds);
t_ds   = torque_output(idx_ds);
ok_ds  = plausibility_ok(idx_ds);

% Colors
c_apps1  = [0.20 0.72 1.00];
c_apps2  = [1.00 0.55 0.10];
c_diff   = [0.85 0.25 0.35];
c_thresh = [1.00 0.90 0.00];
c_torque = [0.25 0.90 0.50];
c_fault  = [0.85 0.25 0.35];
c_grid   = [0.28 0.28 0.32];
bg_color = [0.12 0.12 0.15];
ax_color = [0.17 0.17 0.21];

fig = figure('Name', 'APPS Signal Processing & Torque Output', ...
             'Color', bg_color, 'Position', [80, 60, 1200, 820], 'Visible', 'on');

% ---- Subplot 1 -----------------------------------------------------------
ax1 = subplot(3,1,1);
set(ax1, 'Color', ax_color, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
    'GridColor', c_grid, 'GridAlpha', 0.6, 'FontSize', 10);
hold on; grid on; box on;
plot(s_ds, a1_ds, '-', 'Color', c_apps1, 'LineWidth', 1.4, 'DisplayName', 'APPS1');
plot(s_ds, a2_ds, '-', 'Color', c_apps2, 'LineWidth', 1.4, 'DisplayName', 'APPS2');
yline(P_DEAD, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0, 'HandleVisibility', 'off');
ylabel('Pedal Travel (%)', 'Color', [0.85 0.85 0.85], 'FontSize', 11);
title('Normalized APPS Signals — Pedal Travel', 'Color', [0.95 0.95 0.95], 'FontSize', 12, 'FontWeight', 'bold');
leg1 = legend('Location', 'northeast', 'FontSize', 9);
set(leg1, 'Color', [0.22 0.22 0.26], 'TextColor', [0.9 0.9 0.9], 'EdgeColor', [0.35 0.35 0.4]);
ylim([-5 110]); xlim([s_ds(1) s_ds(end)]);

% ---- Subplot 2 -----------------------------------------------------------
ax2 = subplot(3,1,2);
set(ax2, 'Color', ax_color, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
    'GridColor', c_grid, 'GridAlpha', 0.6, 'FontSize', 10);
hold on; grid on; box on;
fault_ds    = ~ok_ds;
f_starts    = find(diff([0; fault_ds]) == 1);
f_ends      = find(diff([fault_ds; 0]) == -1);
y_top       = max(d_ds) * 1.2;
for k = 1:length(f_starts)
    fill([s_ds(f_starts(k)) s_ds(f_ends(k)) s_ds(f_ends(k)) s_ds(f_starts(k))], ...
         [0 0 y_top y_top], c_fault, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
plot(s_ds, d_ds, '-', 'Color', c_diff, 'LineWidth', 1.4, 'DisplayName', '|APPS1-APPS2|');
yline(PLAUSIBILITY_THRESHOLD, '--', 'Color', c_thresh, 'LineWidth', 1.8, 'DisplayName', '10% Threshold');
ylabel('Difference (%)', 'Color', [0.85 0.85 0.85], 'FontSize', 11);
title('Plausibility Check — |APPS1 - APPS2|  (red shading = fault zone)', ...
      'Color', [0.95 0.95 0.95], 'FontSize', 12, 'FontWeight', 'bold');
leg2 = legend('Location', 'northeast', 'FontSize', 9);
set(leg2, 'Color', [0.22 0.22 0.26], 'TextColor', [0.9 0.9 0.9], 'EdgeColor', [0.35 0.35 0.4]);
ylim([0 max(y_top, PLAUSIBILITY_THRESHOLD*1.4)]); xlim([s_ds(1) s_ds(end)]);

% ---- Subplot 3 -----------------------------------------------------------
ax3 = subplot(3,1,3);
set(ax3, 'Color', ax_color, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
    'GridColor', c_grid, 'GridAlpha', 0.6, 'FontSize', 10);
hold on; grid on; box on;
fill([s_ds; flipud(s_ds)], [t_ds; zeros(length(s_ds),1)], ...
     c_torque, 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(s_ds, t_ds, '-', 'Color', c_torque, 'LineWidth', 1.4, 'DisplayName', 'Motor Torque');
yline(T_MAX, ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.2, 'HandleVisibility', 'off');
xlabel('Sample Index', 'Color', [0.85 0.85 0.85], 'FontSize', 11);
ylabel('Torque (Nm)', 'Color', [0.85 0.85 0.85], 'FontSize', 11);
title(sprintf('Motor Torque Output  [Tmax=%d Nm, Pdead=%.0f%%, n=%d]', T_MAX, P_DEAD, N_EXPONENT), ...
      'Color', [0.95 0.95 0.95], 'FontSize', 12, 'FontWeight', 'bold');
leg3 = legend('Location', 'northeast', 'FontSize', 9);
set(leg3, 'Color', [0.22 0.22 0.26], 'TextColor', [0.9 0.9 0.9], 'EdgeColor', [0.35 0.35 0.4]);
ylim([-5 T_MAX*1.12]); xlim([s_ds(1) s_ds(end)]);

sgtitle('APPS Dual-Channel Signal Processing & Motor Torque Mapping', ...
        'Color', [1 1 1], 'FontSize', 14, 'FontWeight', 'bold');
linkaxes([ax1, ax2, ax3], 'x');
drawnow;

%% -------------------------------------------------------------------------
%  SECTION 7: EXPORT RESULTS
% -------------------------------------------------------------------------

% Save figure as high-resolution PNG
save_path = fullfile(pwd, 'apps_torque_output.png');
print(fig, save_path, '-dpng', '-r150');
fprintf('[OK] Plot saved to: %s\n', save_path);

% Save results table to CSV
results_table = table(sample_idx, raw_apps1, raw_apps2, ...
                      filt_apps1, filt_apps2, abs_diff, ...
                      plausibility_ok, torque_output, ...
    'VariableNames', {'Sample','RawAPPS1_ADC','RawAPPS2_ADC', ...
                      'NormAPPS1_pct','NormAPPS2_pct','AbsDiff_pct', ...
                      'PlausibilityOK','Torque_Nm'});

writetable(results_table, 'apps_torque_results.csv');
fprintf('[OK] Results saved to "apps_torque_results.csv"\n');

fprintf('\n=== Processing Complete ===\n');

%% -------------------------------------------------------------------------
%  LOCAL FUNCTION: Manual Median Filter (no toolbox required)
% -------------------------------------------------------------------------
function out = manual_medfilt(sig, win)
    n   = length(sig);
    out = zeros(n, 1);
    half = floor(win / 2);
    for i = 1:n
        i_start = max(1, i - half);
        i_end   = min(n, i + half);
        out(i)  = median(sig(i_start:i_end));
    end
end