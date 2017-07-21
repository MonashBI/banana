function QSM( inDir, maskFile, outDir, echoTimes )
	switch numel(echoTimes)
		case 2
			disp('Running DualEcho QSM Reconstruction')
			QSM_DualEcho(inDir, maskFile, outDir, echoTimes);
		case 1
			disp('Running SingleEcho QSM Reconstruction')
			QSM_SingleEcho(inDir, maskFile, outDir, echoTimes);
		otherwise
			disp('Invalid echo times provided')
			throw MException('QSM:InvalidEchoTimes','Echo times vector size was not 1 or 2')
	end
end