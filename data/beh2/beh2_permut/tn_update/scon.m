function AA = scon(AA, v, ord, ford)
% rpscon v0.1.2.3 (traces and trivial indices)
% This version by R. Pfeifer, 2009, adapted from an earlier version of the
% scon function by Guifre Vidal. Incorporates con2t by "Frank" (Verstraete?).
%
% AA = {A1, A2, ..., Ap} cell of tensors
% v = {v1, v2, ..., vp} cell of vectors 
% e.g. v1 = [3 4 -1] labels the three indices of tensor A1, with -1 indicating an uncontracted index (open leg) 
% [x 1] tensors receive special treatment
% ord, if present, contains a list of all indices ordered - if not, [1 2 3 4 ..] by default
% ford, if present, contains the final ordering of the uncontracted indices - if not, [-1 -2 ..] by default
% This version: Handles traces on a single tensor
%               Handles trivial indices, including trailing ones (suppressed by Matlab)

% Change list:
% v0.1.2.3: Added support for networks where a disconnected portion reduces to a number, e.g. scon({A,B},{[1 1],[-1 -2]})
% v0.1.2.2: Fixed support for scon({A,B},{[-1 -2],[-3 -4]})
% v0.1.2: Now detects if there are multiple parts left after contracting all positive indices.
%         If so, automatically inserts trivial indices and contracts them.
% v0.1.1: Fixed bug causing crash when output is a number
% v0.1.0: Created from Guifre's scon.m.

check_indices = 1; % (0)1 - (don't) check consistency of indices

if size(AA,1)~=1
    error('Array of tensors has incorrect dimension - should be 1xn')
end

if nargin < 3
    ord = create_ord(v);
end
if nargin < 4 
    ford = create_ford(v);
end                  
if check_indices == 1
    make_check_indices(AA,v,ord); % if there is an error, the program will stop
end
% [w_con, w_unc] = compute_weights(AA, v);
tensorSizes = compute_sizes(AA,v);

while (size(v,2) > 1) || (size(v{1},2) > size(ford,2)) % When on last tensor, check if any traces to perform before leaving loop
    if isempty(ord)
        % If, once all contractions are finished, we have many disjoint parts, connect them using trivial indices.
        ord = 1;
        v{1} = [v{1} 1];
        v{2} = [v{2} 1];
        tensorSizes{1} = [tensorSizes{1} 1];
        tensorSizes{2} = [tensorSizes{2} 1];
    end
    tcon = get_tcon(v,ord(1)); % 'tcon' = tensors to be contracted    
    if size(tcon,2)==1
        tracing = true;
        icon = ord(1); % Only contract one index at a time if tracing
    else
        tracing = false;
        icon = get_icon(v,tcon); % 'icon' = indices to be contracted
    end
    [pos1, pos2] = get_pos(v,[tcon tcon],icon); % position in AA1 and AA2 of indices to be contracted
    % Sneaky: If size of tcon is already 2, elements 3 and 4 are ignored.

%     Q = con2t(AA{tcon(1)}, AA{tcon(2)}, pos1, pos2);
%     Q = squeeze(Q);
% % sAA1=size(AA{tcon(1)}) 
% % sAA2=size(AA{tcon(2)})
% % sAAend = size(Q)
%     AA = {AA{:} Q};
%     AA(tcon) = []; % update list of tensors AA

    newAApos = size(AA,2) + 1; % position in AA to insert new tensor
    if tracing
        [AA{newAApos} tensorSizes{newAApos}] = doTrace(AA{tcon(1)},pos1,tensorSizes{tcon(1)}); % Trace on a tensor % Removed squeeze
    else
        if numel(tensorSizes{tcon(1)})==1
            tensorSizes{tcon(1)} = [tensorSizes{tcon(1)} 1];
        end
        if numel(tensorSizes{tcon(2)})==1
            tensorSizes{tcon(2)} = [tensorSizes{tcon(2)} 1];
        end
        [AA{newAApos} tensorSizes{newAApos}] = con2t(AA{tcon(1)}, AA{tcon(2)}, pos1, pos2, tensorSizes{tcon(1)}, tensorSizes{tcon(2)}); % contraction of 2 tensors % Removed squeeze
    end
    AA(tcon) = []; % update list of tensors AA
    tensorSizes(tcon) = []; % Update list of tensor sizes

% sv1 = v{tcon(1)}
% sv2 = v{tcon(2)}
    v = {v{:} find_newv(v,tcon,icon)}; % Need to pass icon so find_newv doesn't accidentally clobber traces
    v(tcon) = []; % update graph v
% svend = v{end}
    ord = renew_ord(ord, icon); % update ord
    
end

vlast = v{1};

% Qout = AA{1};
% if size(vlast,2)>1
%     Qout = get_Qout(Qout, vlast, ford);
% end

AA = AA{1};

if size(vlast,2)>1
    AA = get_Qout(AA,vlast,ford);
end

% End the function
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Q = get_Qout(Q, v, ford)
perm=[];
for rr = 1:size(ford,2)
    perm = [perm find(v == ford(rr))];
end
Q = permute(Q, perm);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ord = renew_ord(ord, icon); % update ord
for ss = 1:size(icon,2)
    % ord(find(ord == icon(ss))) = [];
    ord((ord == icon(ss))) = []; % Logical indexing is faster - better form. (Not v crucial)
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function newv = find_newv(v,tcon,icon)
if size(tcon,2)==2
    newv = [v{tcon(1)} v{tcon(2)}];
else
    newv = v{tcon(1)};
end
rr=1;
while rr <= size(newv,2) % kill repeated indices that have been summed over
    pos = find(newv == newv(rr));
    summed = find(icon == newv(pos(1)));
    if size(pos,2)>1 && size(summed,2)>0
        newv(pos) = [];
    else 
        rr = rr+1;
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pos1, pos2] = get_pos(v,tcon,icon)
pos1 = [];
pos2 = [];
for rr = 1:size(icon,2)
    pos1 = [pos1 find(v{tcon(1)} == icon(rr))];
    pos2 = [pos2 find(v{tcon(2)} == icon(rr))];
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tcon = get_tcon(v,index)
tcon = [];
for rr = 1:size(v,2)
    if size(find(v{rr} == index),2)>0
        tcon = [tcon rr];
    end
end
if size(tcon,2) > 2
    error(['Wrong number of tensors with index=' num2str(index) 10 'for tcon=' num2str(tcon)])
elseif size(tcon,2) == 1 && size(find(v{tcon(1)}==index),2) ~= 2
    % only one tensor has index, and this is not a trace
    error(['Only one tensor with index ' num2str(index) ' and this is not a trace!'])
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function icon = get_icon(v,tcon)
icon = [];
for rr = 1:size(v{tcon(1)},2)
    if size(find(v{tcon(2)} == v{tcon(1)}(rr)),2) > 0
        icon = [icon v{tcon(1)}(rr)];
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ord = create_ord(v)
x = [];
for ss=1:size(v,2) % identify all positive indices
    for rr = 1:size(v{ss},2)
        if v{ss}(rr) > 0 && size(find(x == v{ss}(rr)),2) == 0
            x = [x v{ss}(rr)];
        end
    end
end
ord = sort(x);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ford = create_ford(v)
x = [];
for ss=1:size(v,2) % identify all positive indices
    for rr = 1:size(v{ss},2)
        if v{ss}(rr) < 0 && size(find(x == v{ss}(rr)),2) == 0
            x = [x v{ss}(rr)];
        end
    end
end
ford = -sort(-x);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [w_con, w_unc] = compute_weights(AA, v)
% 
% p = size(AA,2); % vector of weights
% 
% for pp = 1:p
%     weights{pp} = size(AA{pp});
% end
% rrr=0; %any uncontracted index?
% for rr = 1:size(v,2)  % weights for contracted and uncontracted indices
%     for ss = 1:size(v{rr},2)
%         t = v{rr}(ss);
%         if t > 0
%             if ss<=size(weights{rr},2)
%                 w_con(t) = weights{rr}(ss);
%             else
%                 w_con(t) = 1; % Trailing trivial indices dropped by Matlab
%             end
%         elseif t < 0
%             rrr=1;
%             if ss<=size(weights{rr},2)
%                 w_unc(-t) = weights{rr}(ss);    
%             else
%                 w_unc(-t) = 1; % Trailing trivial indices dropped by Matlab
%             end
%         end
%     end
% end
% if rrr==0
%     w_unc = [];
% end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A reexpand]=con2t(A,B,ma,mb,sa,sb,permQ);   % code by Frank -- comments by me (Guifre)...
% contracts indices ma and mb of A and B and (possibly) permutes some of the
% indices of the resulting tensor

% sa=size(A);sa(size(sa,2)+1:max(ma))=1; % Add as many trivial indices as required by this contraction % Not necessary - sa now passed directly to con2t
na=size(sa,2);Ia=1:na;Ia(ma)=[];
% sa = [a1 a2 a3 a4]; size(sa) = [1 4]; na = 4 number of indices of A; Ia =
% [1 2 3 4]; ma = 2 4; Ia(ma) = [1 3]
% Ia is a list of the indices of A that remain after the contraction
% sb=size(B);sb(size(sb,2)+1:max(mb))=1;
nb=size(sb,2);Ib=1:nb;Ib(mb)=[];
if sa(ma) ~= sb(mb) 
    error(['index dimensions do not match' 10 'Indices ' num2str(ma) ' & ' num2str(mb)])
end
A=permute(A,[Ia ma]);
% reorganize the indices according to: "not to be contracted" and "to be contracted"
B=permute(B,[Ib mb]);
A=reshape(A,[prod(sa(Ia)) prod(sa(ma))])*reshape(B,[prod(sb(Ib)) prod(sb(mb))]).';
% reshape the tensors into matrices, perform a matrix product
if size(sa(Ia),2) > 0 || size(sb(Ib),2) > 0 
    reexpand = [sa(Ia) sb(Ib)];    
    A=reshape(A,[reexpand 1]);
    % reexpand the remaning indices
else
    reexpand = 1;
end
if nargin==5,
    A=permute(A,permQ);
end
% possibly permute final indices
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [B reexpand]=doTrace(A,ma,sa)
%sa=size(A);
na=size(sa,2);Ia=1:na;Ia(ma)=[];
if sa(ma(1))~=sa(ma(2))
    error(['index dimensions do not match' 10 'Indices ' num2str(ma) ' & ' num2str(mb)])
end
A=reshape(permute(A,[Ia ma]),prod(sa(Ia)),prod(sa(ma))); % Separate indices to be traced and not to be traced
B=0;
for aa=1:sa(ma(2))
    B=B+A(:,aa+(aa-1)*sa(ma(2))); % Perform trace
end
clear A
reexpand = sa(Ia);
if size(reexpand,2) > 0
    B=reshape(B,[reexpand 1]); % Restore remaining indices; trailing 1 needed if sa(Ia) empty)
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function make_check_indices(AA,v,ord)
p = size(AA,2);

for pp=1:p
    dAA{pp} = size(AA{pp});
    
    % Allow trivial indices
    % if size(find(dAA{pp} == 1),2)>0 %% any index of rank 1?
    %     whereones = find(dAA{pp} == 1); 
    %     if size(find(whereones ~= 2),2) >0 %% is it not in the second position ?
    %         display(['trivial index for AA(pp)' 10 'pp=' num2str(pp) 10 'dAApp=' num2str(dAA{pp})])
    %     end
    % end
    if size(dAA{pp},2) ~= size(v{pp},2)
        % if size(dAA{pp},2) ~= 2 || size(v{pp},2) ~= 1
        %     error(['error: dimensions do not match in scon!!!!!!' 10 'pp: ' num2str(pp)]);
        % end
        % This error used to be thrown up if there was a trailing trivial index.
        for mm=1:size(v{pp},2)
            for ppp=pp+1:p
                coinci = find(v{ppp} == v{pp}(mm));
                if size(coinci,2) > 0
                    if size(AA{ppp}, coinci) ~= size(AA{pp},mm)
                        error(['error: dimensions do not match II in scon AA{ppp}, AA{pp}(mm)!!!!!!!' 10 'ppp: ' num2str(ppp) 10 'coinci: ' num2str(coinci) 10 'pp: ' num2str(pp) 10 'mm: ' num2str(mm)])
                    end
                end                   
            end
        end
    end
end
end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tS = compute_sizes(AA,v)
% Creates arrays holding size(AA{a}), INCLUDING any implicit trailing dimensions of size 1 or 0.

numTensors = size(AA,2);
tS = cell(1,numTensors);
for a=1:numTensors
    tS{a} = size(AA{a});
    if size(v{a},2) > size(tS{a},2)
        tS{a} = [tS{a} ones(1,size(v{a},2)-size(tS{a},2))];
    end
end
end
