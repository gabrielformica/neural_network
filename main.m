source('backpropagation.m')

function main(training_file)
    Wih = rand(2,10) - 0.5
    Who = rand(10,1) - 0.5
    for j = 1 : 1000
       [xs,ys,ts] = textread(training_file, '%f %f %f', 'delimiter', ' ');
       for i = 1:length(xs)
           target = ts(i);
           if (target < 0)
              target = 0;
           endif
           [Wih, Who] = backpropagation([xs(i), ys(i)], target, Wih, Who, 0.01);
       endfor
    endfor

    test('set500.txt', Wih, Who)
endfunction

function test(testing_file, Wih, Who)
    [xs,ys,ts] = textread(testing_file, '%f %f %f', 'delimiter', ' ');
    for i = 1 : length(xs)
        target = ts(i);
        if (target < 0)
            target = 0; 
        endif
        [valuesh, output, deltao] = feedforward(Wih, Who, [xs(i), ys(i)], target)
        if (output >= 0.5)
           plot(xs(i), ys(i), 'g+')
           hold on
        else 
           plot(xs(i), ys(i), 'r+')
           hold on
        endif
    endfor
endfunction
