% =========================================================================
% APPS Signal Processing, Motor Torque & Regenerative Braking
% Accelerator Pedal Position Sensor (APPS) Dual-Channel Safety System
% =========================================================================

clc; clear; close all;

%% SECTION 1: CONFIGURATION & CONSTANTS

ADC_RESOLUTION  = 4095;
V_REF           = 3.3;
V_PER_COUNT     = V_REF / ADC_RESOLUTION;

APPS1_ADC_MIN   = 0;
APPS1_ADC_MAX   = round(1.68 / V_PER_COUNT);   % ~2084
APPS2_ADC_MIN   = round(1.70 / V_PER_COUNT);   % ~2109
APPS2_ADC_MAX   = ADC_RESOLUTION;

T_MAX                  = 230;    % Max acceleration torque (Nm)
P_DEAD                 = 5.0;    % Deadband (%)
N_EXPONENT             = 2;      % Quadratic map exponent
PLAUSIBILITY_THRESHOLD = 10.0;   % Max APPS difference before fault (%)
MEDIAN_FILTER_WIN      = 5;      % Median filter window

% --- Regen Braking Parameters (matching Simulink block settings) ---
BRAKE_STEP_TIME  = 5.0;    % Step time: brake pressed at 5s
BRAKE_INIT_VAL   = 0.0;    % Initial value: 0%
BRAKE_FINAL_VAL  = 40.0;   % Final value: 40%
REGEN_GAIN       = -0.8;   % Gain: 100% brake → -80 Nm  (Gain = -0.8)
TORQUE_SAT_MAX   = 200.0;  % Saturation upper limit: 200 Nm
TORQUE_SAT_MIN   = -80.0;  % Saturation lower limit: -80 Nm

fprintf('=== APPS Torque + Regen Processor ===\n');
fprintf('Regen Gain      : %.1f  (100%% brake = %.0f Nm)\n', REGEN_GAIN, 100*REGEN_GAIN);
fprintf('Saturation      : [%.0f Nm  to  %.0f Nm]\n\n', TORQUE_SAT_MIN, TORQUE_SAT_MAX);

%% SECTION 2: DATA INPUT

apps1_file = 'apps1_app.txt';
apps2_file = 'apps2_app.txt';

if isfile(apps1_file)
    raw_apps1 = load(apps1_file);
    fprintf('[OK] Loaded APPS1: %d samples\n', length(raw_apps1));
else
    warning('apps1_app.txt not found — using synthetic data.');
    pedal_profile = [linspace(0,1,500)'; ones(500,1); linspace(1,0.3,400)'; ...
                     linspace(0.3,0.8,300)'; linspace(0.8,0,300)'];
    raw_apps1 = round(APPS1_ADC_MIN + pedal_profile.*(APPS1_ADC_MAX-APPS1_ADC_MIN) + randn(2000,1)*8);
    raw_apps1 = max(0, min(ADC_RESOLUTION, raw_apps1));
end

if isfile(apps2_file)
    raw_apps2 = load(apps2_file);
    fprintf('[OK] Loaded APPS2: %d samples\n', length(raw_apps2));
else
    warning('apps2_app.txt not found — using synthetic data.');
    pedal_profile2 = pedal_profile;
    pedal_profile2(800:850) = pedal_profile2(800:850) + 0.18;
    raw_apps2 = round(APPS2_ADC_MIN + pedal_profile2.*(APPS2_ADC_MAX-APPS2_ADC_MIN) + randn(2000,1)*8);
    raw_apps2 = max(0, min(ADC_RESOLUTION, raw_apps2));
end

n_samples  = min(length(raw_apps1), length(raw_apps2));
raw_apps1  = raw_apps1(1:n_samples);
raw_apps2  = raw_apps2(1:n_samples);
sample_idx = (1:n_samples)';
fprintf('Processing %d aligned samples.\n\n', n_samples);

%% SECTION 3: NORMALIZATION

clamped_apps1 = max(APPS1_ADC_MIN, min(APPS1_ADC_MAX, raw_apps1));
clamped_apps2 = max(APPS2_ADC_MIN, min(APPS2_ADC_MAX, raw_apps2));
norm_apps1 = ((clamped_apps1 - APPS1_ADC_MIN) / (APPS1_ADC_MAX - APPS1_ADC_MIN)) * 100;
norm_apps2 = ((clamped_apps2 - APPS2_ADC_MIN) / (APPS2_ADC_MAX - APPS2_ADC_MIN)) * 100;

%% SECTION 4: MEDIAN FILTER

filt_apps1 = manual_medfilt(norm_apps1, MEDIAN_FILTER_WIN);
filt_apps2 = manual_medfilt(norm_apps2, MEDIAN_FILTER_WIN);
abs_diff   = abs(filt_apps1 - filt_apps2);

fprintf('Signal Statistics (filtered):\n');
fprintf('  APPS1 — Min: %.2f%%  Max: %.2f%%  Mean: %.2f%%\n', min(filt_apps1), max(filt_apps1), mean(filt_apps1));
fprintf('  APPS2 — Min: %.2f%%  Max: %.2f%%  Mean: %.2f%%\n', min(filt_apps2), max(filt_apps2), mean(filt_apps2));
fprintf('  |Diff| — Max: %.2f%%  Mean: %.2f%%\n\n', max(abs_diff), mean(abs_diff));

%% SECTION 5: ACCELERATION TORQUE (plausibility-gated quadratic map)

pedal_avg        = (filt_apps1 + filt_apps2) / 2;
plausibility_ok  = abs_diff <= PLAUSIBILITY_THRESHOLD;

torque_accel     = zeros(n_samples, 1);
active           = pedal_avg > P_DEAD;
travel_norm      = (pedal_avg(active) - P_DEAD) / (100 - P_DEAD);
torque_accel(active) = T_MAX .* (travel_norm) .^ N_EXPONENT;
torque_accel     = torque_accel .* plausibility_ok;   % zero on fault

n_faults  = sum(~plausibility_ok);
fprintf('Plausibility Faults : %d / %d samples (%.1f%%)\n', n_faults, n_samples, (n_faults/n_samples)*100);
fprintf('Peak Accel Torque   : %.2f Nm\n\n', max(torque_accel));

%% SECTION 6: REGENERATIVE BRAKING

% --- Step signal: 0% until BRAKE_STEP_TIME, then BRAKE_FINAL_VAL% ---
% Time axis uses same sample rate as the data (sample index = time proxy)
% We map sample index → time using fs = 100 Hz assumption
fs_assumed = 100;   % Hz — change if your data has a different rate
time_vec   = (sample_idx - 1) / fs_assumed;   % seconds

% Step block equivalent: Initial=0, Final=40, StepTime=5s
brake_pct  = BRAKE_INIT_VAL * ones(n_samples, 1);
brake_pct(time_vec >= BRAKE_STEP_TIME) = BRAKE_FINAL_VAL;

% Gain block: Brake% × -0.8 → Tregen
T_regen    = brake_pct * REGEN_GAIN;   % e.g. 40% × -0.8 = -32 Nm

% Sum block: Taccel + Tregen
torque_combined = torque_accel + T_regen;

% Saturation block: clamp to [TORQUE_SAT_MIN, TORQUE_SAT_MAX]
torque_final = max(TORQUE_SAT_MIN, min(TORQUE_SAT_MAX, torque_combined));

fprintf('Regen Stats:\n');
fprintf('  Brake step applied at t = %.1f s (sample %d)\n', BRAKE_STEP_TIME, find(time_vec >= BRAKE_STEP_TIME, 1));
fprintf('  Regen torque at 40%% brake : %.1f Nm\n', BRAKE_FINAL_VAL * REGEN_GAIN);
fprintf('  Peak Final Torque  : %.2f Nm\n', max(torque_final));
fprintf('  Min  Final Torque  : %.2f Nm\n\n', min(torque_final));

%% SECTION 7: PLOTTING (downsampled for speed)

PLOT_EVERY = max(1, floor(n_samples / 5000));
idx_ds = 1:PLOT_EVERY:n_samples;
fprintf('Plotting %d points (1 in every %d)...\n', length(idx_ds), PLOT_EVERY);

s_ds    = sample_idx(idx_ds);
a1_ds   = filt_apps1(idx_ds);
a2_ds   = filt_apps2(idx_ds);
d_ds    = abs_diff(idx_ds);
ok_ds   = plausibility_ok(idx_ds);
ta_ds   = torque_accel(idx_ds);
br_ds   = brake_pct(idx_ds);
tr_ds   = T_regen(idx_ds);
tf_ds   = torque_final(idx_ds);

bg  = [0.12 0.12 0.15];
ax_c= [0.17 0.17 0.21];
cA1 = [0.20 0.72 1.00];
cA2 = [1.00 0.55 0.10];
cDf = [0.85 0.25 0.35];
cTh = [1.00 0.90 0.00];
cAc = [0.25 0.90 0.50];
cRg = [0.90 0.40 1.00];
cFn = [1.00 0.85 0.20];
cGr = [0.28 0.28 0.32];

fig = figure('Name','APPS + Regen Torque','Color',bg,'Position',[60 40 1280 960],'Visible','on');

% ---- Subplot 1: APPS Signals ---------------------------------------------
ax1 = subplot(4,1,1);
set(ax1,'Color',ax_c,'XColor',[0.7 0.7 0.7],'YColor',[0.7 0.7 0.7],'GridColor',cGr,'GridAlpha',0.6,'FontSize',9);
hold on; grid on; box on;
plot(s_ds, a1_ds, '-', 'Color', cA1, 'LineWidth', 1.4, 'DisplayName', 'APPS1');
plot(s_ds, a2_ds, '-', 'Color', cA2, 'LineWidth', 1.4, 'DisplayName', 'APPS2');
yline(P_DEAD,'--','Color',[0.5 0.5 0.5],'LineWidth',1.0,'HandleVisibility','off');
ylabel('Pedal (%)','Color',[0.85 0.85 0.85],'FontSize',10);
title('Normalized APPS1 & APPS2 — Pedal Travel','Color',[0.95 0.95 0.95],'FontSize',11,'FontWeight','bold');
leg1 = legend('Location','northeast','FontSize',8);
set(leg1,'Color',[0.2 0.2 0.25],'TextColor',[0.9 0.9 0.9],'EdgeColor',[0.35 0.35 0.4]);
ylim([-5 110]); xlim([s_ds(1) s_ds(end)]);

% ---- Subplot 2: Plausibility Difference ----------------------------------
ax2 = subplot(4,1,2);
set(ax2,'Color',ax_c,'XColor',[0.7 0.7 0.7],'YColor',[0.7 0.7 0.7],'GridColor',cGr,'GridAlpha',0.6,'FontSize',9);
hold on; grid on; box on;
fault_ds = ~ok_ds;
f_starts = find(diff([0; fault_ds]) == 1);
f_ends   = find(diff([fault_ds; 0]) == -1);
y_top    = max(d_ds) * 1.2;
for k = 1:length(f_starts)
    fill([s_ds(f_starts(k)) s_ds(f_ends(k)) s_ds(f_ends(k)) s_ds(f_starts(k))], ...
         [0 0 y_top y_top], cDf, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
plot(s_ds, d_ds, '-', 'Color', cDf, 'LineWidth', 1.4, 'DisplayName', '|APPS1-APPS2|');
yline(PLAUSIBILITY_THRESHOLD,'--','Color',cTh,'LineWidth',1.8,'DisplayName','10% Limit');
ylabel('Diff (%)','Color',[0.85 0.85 0.85],'FontSize',10);
title('Plausibility Check — Fault zones shaded red','Color',[0.95 0.95 0.95],'FontSize',11,'FontWeight','bold');
leg2 = legend('Location','northeast','FontSize',8);
set(leg2,'Color',[0.2 0.2 0.25],'TextColor',[0.9 0.9 0.9],'EdgeColor',[0.35 0.35 0.4]);
ylim([0 max(y_top, PLAUSIBILITY_THRESHOLD*1.5)]); xlim([s_ds(1) s_ds(end)]);

% ---- Subplot 3: Accel + Regen torques separately -------------------------
ax3 = subplot(4,1,3);
set(ax3,'Color',ax_c,'XColor',[0.7 0.7 0.7],'YColor',[0.7 0.7 0.7],'GridColor',cGr,'GridAlpha',0.6,'FontSize',9);
hold on; grid on; box on;
plot(s_ds, ta_ds, '-', 'Color', cAc, 'LineWidth', 1.4, 'DisplayName', 'T_{accel}');
plot(s_ds, tr_ds, '-', 'Color', cRg, 'LineWidth', 1.4, 'DisplayName', 'T_{regen}');
yline(0,'--','Color',[0.5 0.5 0.5],'LineWidth',0.8,'HandleVisibility','off');
ylabel('Torque (Nm)','Color',[0.85 0.85 0.85],'FontSize',10);
title('Acceleration Torque vs Regenerative Brake Torque','Color',[0.95 0.95 0.95],'FontSize',11,'FontWeight','bold');
leg3 = legend('Location','northeast','FontSize',8);
set(leg3,'Color',[0.2 0.2 0.25],'TextColor',[0.9 0.9 0.9],'EdgeColor',[0.35 0.35 0.4]);
ylim([TORQUE_SAT_MIN*1.2  T_MAX*1.1]); xlim([s_ds(1) s_ds(end)]);

% ---- Subplot 4: Final Saturated Torque -----------------------------------
ax4 = subplot(4,1,4);
set(ax4,'Color',ax_c,'XColor',[0.7 0.7 0.7],'YColor',[0.7 0.7 0.7],'GridColor',cGr,'GridAlpha',0.6,'FontSize',9);
hold on; grid on; box on;
fill([s_ds; flipud(s_ds)], [tf_ds; zeros(length(s_ds),1)], cFn, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(s_ds, tf_ds, '-', 'Color', cFn, 'LineWidth', 1.6, 'DisplayName', 'Final Torque');
yline(TORQUE_SAT_MAX, ':', 'Color', cAc, 'LineWidth', 1.2, 'DisplayName', sprintf('+%.0f Nm limit', TORQUE_SAT_MAX));
yline(TORQUE_SAT_MIN, ':', 'Color', cRg, 'LineWidth', 1.2, 'DisplayName', sprintf('%.0f Nm limit', TORQUE_SAT_MIN));
yline(0,'--','Color',[0.5 0.5 0.5],'LineWidth',0.8,'HandleVisibility','off');
xlabel('Sample Index','Color',[0.85 0.85 0.85],'FontSize',10);
ylabel('Torque (Nm)','Color',[0.85 0.85 0.85],'FontSize',10);
title(sprintf('Final Motor Torque  [Saturated: %.0f to %.0f Nm]', TORQUE_SAT_MIN, TORQUE_SAT_MAX), ...
      'Color',[0.95 0.95 0.95],'FontSize',11,'FontWeight','bold');
leg4 = legend('Location','northeast','FontSize',8);
set(leg4,'Color',[0.2 0.2 0.25],'TextColor',[0.9 0.9 0.9],'EdgeColor',[0.35 0.35 0.4]);
ylim([TORQUE_SAT_MIN*1.3  TORQUE_SAT_MAX*1.15]); xlim([s_ds(1) s_ds(end)]);

sgtitle('APPS Dual-Channel Processing + Regenerative Braking', ...
        'Color',[1 1 1],'FontSize',13,'FontWeight','bold');
linkaxes([ax1 ax2 ax3 ax4], 'x');
drawnow;

%% SECTION 8: EXPORT

save_path = fullfile(pwd, 'apps_torque_output.png');
print(fig, save_path, '-dpng', '-r150');
fprintf('[OK] Plot saved to: %s\n', save_path);

results_table = table(sample_idx, filt_apps1, filt_apps2, abs_diff, ...
                      plausibility_ok, torque_accel, brake_pct, T_regen, torque_final, ...
    'VariableNames', {'Sample','NormAPPS1_pct','NormAPPS2_pct','AbsDiff_pct', ...
                      'PlausibilityOK','AccelTorque_Nm','BrakePct', ...
                      'RegenTorque_Nm','FinalTorque_Nm'});
writetable(results_table, 'apps_torque_results.csv');
fprintf('[OK] Results saved to: apps_torque_results.csv\n');
fprintf('\n=== Processing Complete ===\n');

%% LOCAL FUNCTION: Manual Median Filter
function out = manual_medfilt(sig, win)
    n    = length(sig);
    out  = zeros(n, 1);
    half = floor(win / 2);
    for i = 1:n
        i_start = max(1, i - half);
        i_end   = min(n, i + half);
        out(i)  = median(sig(i_start:i_end));
    end
end