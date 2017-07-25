function c = trackdistance(t1, t2, ep)
	%Compute the distance between two tracks with an unassigned/lost penalty of 10
	if (nargin < 3) ep = 10; end
	l1 = size(t1,1);
	l2 = size(t2,1);
	%Take care of special cases
	if l1 == 0 & l2 > 0
		c = ep*size(t2,1);
	elseif l1 > 0 & l2 == 0
		c = ep*size(t1,1);
	elseif l1 == 0 & l2 == 0
		c = 0;
	%Then the general case
	else 
		%Find overlapping times...
		[C,i1,i2] = intersect(t1(:,1),t2(:,1));
		matched = size(C,1);
		p1 = t1(i1,2:3);
		p2 = t2(i2,2:3);
		d = sqrt(sum((p1-p2).^2,2));
		c = sum(min(d,ep));
		%Non overlapping times get an 'ep' penalty
		c = c + ep*(l1-matched) + ep*(l2-matched);
	end
end