wait[process_,prompt_]:=Block[{str={""}},Do[Pause[0.05];If[!TrueQ[ProcessStatus[process]=="Running"],str={"",EndOfFile,""};Break[]];str=str~Join~StringSplit[ReadString[process,EndOfBuffer],"\n"];If[StringMatchQ[Last@str,prompt~~___],Break[]];,\[Infinity]];str[[2;;-2]]];
sendandwait[process_,command_,prompt_]:=(WriteLine[process,command];wait[process,prompt])
boardempties[board_]:=StringCount[board,"-"];
edaxsearchboard[board_]:=(sendandwait[edax,"setboard "<>board,">"];
edaxstr=sendandwait[edax,"go",">"];
StringJoin@Flatten@StringCases[edaxstr,"Edax plays "~~square__~~EndOfString:>square])
neuralsearchboard[board_]:=(sendandwait[neural,"setboard "<>board,">"];
StringJoin[neuralstr=sendandwait[neural,"go",">"]])
neuralupdateboard[board_,move_]:=(sendandwait[neural,"setboard "<>board,">"];sendandwait[neural,"move "<>move,">"];sendandwait[neural,"textform "<>move,">"][[1]])
neuralgeteval[board_]:=(sendandwait[neural,"setboard "<>board,">"];sendandwait[neural,"eval",">"])
neural=StartProcess[{"./othello","--mode","interactive","--net-path","run5/nets/net-0188","--playouts","33000"}];
edax=StartProcess[{"./lEdax-x64","-vv","-cpu","-book-usage","off","-level","10","-n-tasks","1"},ProcessDirectory->"/home/tlu/Documents/git/edax-reversi/bin"];wait[edax,">"];wait[neural,">"];
textboard0=RandomChoice@StringSplit[Import["openings.txt","Text"],"\n"];
ismove[move_]:=Head[move]===String&&StringMatchQ[move,RegularExpression["[A-Ha-h][1-8]"]];
Do[
textboard=textboard0;
movehistory={};
result=Null;
Do[If[i>1||neuralmovesfirst,
textmove=neuralsearchboard[textboard];
If[(!ismove[textmove])&&(!ismove[Last@movehistory]),result={999};Break[];];
AppendTo[movehistory,textmove];
textboard=neuralupdateboard[textboard,textmove];
If[boardempties[textboard]<=6,result=neuralgeteval[textboard];Break[]];];
textmove=edaxsearchboard[textboard];
If[(!ismove[textmove])&&(!ismove[Last@movehistory]),result={999};Break[];];
AppendTo[movehistory,textmove];
textboard=neuralupdateboard[textboard,textmove];
If[boardempties[textboard]<=6,result=neuralgeteval[textboard];Break[]];,{i,60}];
Print[StringRiffle[movehistory," "]<>" "<>ToString[If[OddQ@Length@movehistory,-1,1]*If[neuralmovesfirst,1,-1]*ToExpression@result[[1]]]];
,{neuralmovesfirst,{True,False}}];
sendandwait[edax,"quit",">"];
sendandwait[neural,"quit",">"];
