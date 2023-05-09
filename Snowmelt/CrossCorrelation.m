%% 2017 Correlations
jemez2017 = importdata('jemez_spring_2017.txt');
lajara2017 = importdata('lajara_spring_2017.txt');

% deleting DC offset
jemez2017 = jemez2017 - mean(jemez2017);
lajara2017 = lajara2017 - mean(lajara2017);
xx = linspace(0,length(jemez2017));
%figure;
%plot(jemez2017)
%yyaxis right
%hold on
%plot(lajara2017)
%yyaxis left

% Calculate cross-correlation and lag time
[xc,lags] = xcorr(lajara2017, jemez2017, 'coeff');
[maxx, idx] = max(xc); % peak correlation coeff and the ID of the time lag does it occur?
lag_time = lags(idx); % this is my time lag
% Plot cross-correlation function
figure;
subplot(3,2,1)
plot(lags, xc)
xlabel('Lag time (days)')
ylabel('Cross-correlation')
title(['2017. Lag = ', num2str(lag_time), ' days. Peak =' , num2str(maxx)])

%% 2018 Correlations
jemez2018 = importdata('jemez_spring_2018.txt');
lajara2018 = importdata('lajara_spring_2018.txt');
jemez2018 = jemez2018 - mean(jemez2018);
lajara2018 = lajara2018 - mean(lajara2018);
% Calculate cross-correlation and lag time
[xc,lags] = xcorr(lajara2018, jemez2018, 'coeff');
[maxx, idx] = max(xc); % peak correlation coeff and the ID of the time lag does it occur?
lag_time = lags(idx); % this is my time lag
% Plot cross-correlation function
subplot(3,2,2)
plot(lags, xc)
xlabel('Lag time (days)')
ylabel('Cross-correlation')
title(['2018. Lag = ', num2str(lag_time), ' days. Peak =' , num2str(maxx)])

%% 2019 Correlations
jemez2019 = importdata('jemez_spring_2019.txt');
lajara2019 = importdata('lajara_spring_2019.txt');
jemez2019 = jemez2019 - mean(jemez2019);
lajara2019 = lajara2019 - mean(lajara2019);
% Calculate cross-correlation and lag time
[xc,lags] = xcorr(lajara2019, jemez2019, 'coeff');
[maxx, idx] = max(xc); % peak correlation coeff and the ID of the time lag does it occur?
lag_time = lags(idx); % this is my time lag
% Plot cross-correlation function
subplot(3,2,3)
plot(lags, xc)
xlabel('Lag time (days)')
ylabel('Cross-correlation')
title(['2019. Lag = ', num2str(lag_time), ' days. Peak =' , num2str(maxx)])

%% 2020 Correlations
jemez2020 = importdata('jemez_spring_2020.txt');
lajara2020 = importdata('lajara_spring_2020.txt');
jemez2020 = jemez2020 - mean(jemez2020);
lajara2020 = lajara2020 - mean(lajara2020);
% Calculate cross-correlation and lag time
[xc,lags] = xcorr(lajara2020, jemez2020, 'coeff');
[maxx, idx] = max(xc); % peak correlation coeff and the ID of the time lag does it occur?
lag_time = lags(idx); % this is my time lag
% Plot cross-correlation function
subplot(3,2,4)
plot(lags, xc)
xlabel('Lag time (days)')
ylabel('Cross-correlation')
title(['2020. Lag = ', num2str(lag_time), ' days. Peak =' , num2str(maxx)])

%% 2021 Correlations
jemez2021 = importdata('jemez_spring_2021.txt');
lajara2021 = importdata('lajara_spring_2021.txt');
jemez2021 = jemez2021 - mean(jemez2021);
lajara2021 = lajara2021 - mean(lajara2021);
% Calculate cross-correlation and lag time
[xc,lags] = xcorr(lajara2021, jemez2021, 'coeff');
[maxx, idx] = max(xc); % peak correlation coeff and the ID of the time lag does it occur?
lag_time = lags(idx); % this is my time lag
% Plot cross-correlation function
subplot(3,2,5)
plot(lags, xc)
xlabel('Lag time (days)')
ylabel('Cross-correlation')
title(['2020. Lag = ', num2str(lag_time), ' days. Peak =' , num2str(maxx)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2017
figure;
subplot(3,2,1)
scatter(lajara2017,jemez2017, 'Marker','.')
% Calculate correlation coefficient and coefficient of determination
r = corrcoef(lajara2017,jemez2017);
r_squared = r(1,2)^2;
% Add trendline to the plot
p = polyfit(lajara2017,jemez2017,1);
y_fit = polyval(p,lajara2017);
hold on
xlabel('La Jara (cfs)')
ylabel('Jemez (cfs)')
plot(lajara2017,y_fit,'r--')
title(['2017 r = ', num2str(r(1,2)), '. R^2 =' , num2str(r_squared)])
%% 2018
subplot(3,2,2)
scatter(lajara2018,jemez2018, 'Marker','.')
% Calculate correlation coefficient and coefficient of determination
r = corrcoef(lajara2018,jemez2018);
r_squared = r(1,2)^2;
% Add trendline to the plot
p = polyfit(lajara2018,jemez2018,1);
y_fit = polyval(p,lajara2018);
hold on
xlabel('La Jara (cfs)')
ylabel('Jemez (cfs)')
plot(lajara2018,y_fit,'r--')
title(['2018 r = ', num2str(r(1,2)), '. R^2 =' , num2str(r_squared)])
%% 2019
subplot(3,2,3)
scatter(lajara2019,jemez2019, 'Marker','.')
% Calculate correlation coefficient and coefficient of determination
r = corrcoef(lajara2019,jemez2019);
r_squared = r(1,2)^2;
% Add trendline to the plot
p = polyfit(lajara2019,jemez2019,1);
y_fit = polyval(p,lajara2019);
hold on
xlabel('La Jara (cfs)')
ylabel('Jemez (cfs)')
plot(lajara2019,y_fit,'r--')
title(['2019 r = ', num2str(r(1,2)), '. R^2 =' , num2str(r_squared)])
%% 2020
subplot(3,2,4)
scatter(lajara2020,jemez2020, 'Marker','.')
% Calculate correlation coefficient and coefficient of determination
r = corrcoef(lajara2020,jemez2020);
r_squared = r(1,2)^2;
% Add trendline to the plot
p = polyfit(lajara2020,jemez2020,1);
y_fit = polyval(p,lajara2020);
hold on
xlabel('La Jara (cfs)')
ylabel('Jemez (cfs)')
plot(lajara2020,y_fit,'r--')
title(['2020 r = ', num2str(r(1,2)), '. R^2 =' , num2str(r_squared)])
%% 2021
subplot(3,2,5)
scatter(lajara2021,jemez2021, 'Marker','.')
% Calculate correlation coefficient and coefficient of determination
r = corrcoef(lajara2021,jemez2021);
r_squared = r(1,2)^2;
% Add trendline to the plot
p = polyfit(lajara2021,jemez2021,1);
y_fit = polyval(p,lajara2021);
hold on
xlabel('La Jara (cfs)')
ylabel('Jemez (cfs)')
plot(lajara2021,y_fit,'r--')
title(['2021 r = ', num2str(r(1,2)), '. R^2 =' , num2str(r_squared)])