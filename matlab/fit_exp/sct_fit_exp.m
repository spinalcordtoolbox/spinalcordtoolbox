function coeffs = sct_fit_exp(img,TE,verbose)
% coeffvals = sct_fit_exp(img,TE,verbose)
% coeffvals = [I Ti]
% I*exp(-t/Ti)

dims=size(img);
coeffvals=cell(dims(1));


for Z=1:dims(3)
    img_max=max(max(max(max(img(:,:,Z,:)))));
    parfor X=1:dims(1)
        for Y=1:dims(2)
            
            % Set up fittype and options.
            ft = fittype( 'exp1' );
            opts = fitoptions( ft );
            opts.Algorithm = 'Trust-Region';
            opts.Display = 'Off';
            opts.Robust = 'Bisquare';
            
% opts.Lower = [-Inf -0.1 -Inf -0.1];
% opts.StartPoint = [5004 -0.01 500 -0.0081];
% opts.Upper = [Inf Inf Inf Inf];
            
            [xData, yData] = prepareCurveData( TE', squeeze(img(X,Y,Z,:)) );
            
            % Fit model to data.
            [fitresult, gof] = fit( xData, yData, ft, opts );
            coeff_tmp = coeffvalues(fitresult);
            coeff_tmp(2) = -1/coeff_tmp(2);
%             coeff_tmp(4) = -1/coeff_tmp(4);
            if verbose
                disp('      I       Ti  ')
                disp(coeff_tmp)
            end
            coeffvals{X}(Y,Z,:) = coeff_tmp;
            if verbose
                % Plot fit with data.
                h = figure(64);
                plot( fitresult, xData, yData, 'X' );
                % Label axes
                xlabel( 'TE' );
                ylabel( 'I' );
                ylim([0 img_max]); xlim([0 max(TE)]);
                grid on
                filename = ['plots/X' num2str(X) 'Y' num2str(Y) 'Z' num2str(Z) '.png'];
                iSaveplotX( h, filename )
            end
        end
    end
end

coeffs = cell2mat(coeffvals);
coeffs=reshape(coeffs,[dims(1:3),2]);

function iSaveplotX( h, filename )
set(gcf, 'InvertHardCopy', 'off');
if ~exist('plots','dir'), mkdir('plots'); end
print(h,filename,'-dpng');