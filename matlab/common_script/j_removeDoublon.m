function matrix_new = j_removeDoublon(vector)

k=1;
for i=1:length(vector)
    doublon = 0;
    for j=1:i-1
        if vector(j)==vector(i)
            doublon = 1;
        end
    end
    if ~doublon
        matrix_new(k)=vector(i);
        k=k+1;
    end
end
