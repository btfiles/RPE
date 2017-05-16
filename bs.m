function rmse = bs(x, y)
disp(size(x));
disp(size(y));

err = bsxfun(@minus, x, y);
rmse = squeeze(sqrt(mean(err.^2, 1)));