import numpy as np

class RelaxationLabeling(object):

    def __init__(self, compatibility, save=False, UseMatrixMultiplication=True, iterations=50):
        self.compatibility = np.asarray(compatibility, dtype=float)
        self.save = save
        self.useMatrixMultiplication = UseMatrixMultiplication
        self.iterations = iterations

        if self.compatibility.ndim not in (4, 6):
            raise ValueError("compatibility must have 4 or 6 dimensions")
        if self.compatibility.shape[0] == 0 or self.compatibility.shape[1] == 0:
            raise ValueError("compatibility must contain at least one object and label")
        self.numObjects = self.compatibility.shape[0]
        self.numLabels = self.compatibility.shape[1]
        expected = (self.numObjects, self.numLabels) * (self.compatibility.ndim // 2)
        if self.compatibility.shape != expected:
            raise ValueError(
                "compatibility axes must alternate objects and labels; "
                "expected {}, got {}".format(expected, self.compatibility.shape)
            )
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        self.printMatrices = False
        if self.compatibility.ndim == 4:
            self.compatType = 2
        else:
            self.compatType = 3

        self.normalizeSupportAcrossEntireMatrix = False
        self.supportFactor = 1.0
        self.iteration = 0

        self.main()

    def initStrengthAndSupport(self):
        self.strength = np.ones(shape = [self.numObjects, self.numLabels])*1/self.numLabels
        self.support = np.zeros(shape = [self.numObjects, self.numLabels])
        if self.printMatrices:
            print('Initial Support')
            print(self.support)
            print('Initial Strength')
            print(self.strength)

    def updateSupport(self):
        if self.compatType == 2:
            for i in range(self.numObjects):
                for j in range(self.numLabels):
                    self.support[i,j] = 0.0
                    for k in range(self.numObjects):
                        for l in range(self.numLabels):
                            self.support[i,j] += self.strength[k,l]*self.compatibility[i,j,k,l]
                self.normalizeSupport(i)
        if self.compatType == 3: 
            if self.useMatrixMultiplication:
                self.support = np.einsum(
                    "kl,mn,ijklmn->ij",
                    self.strength,
                    self.strength,
                    self.compatibility,
                    optimize=True,
                )
                self.normalizeSupport()
            else:
                self.support.fill(0.0)
                for i in range(self.numObjects):
                    for j in range(self.numLabels):
                        for k in range(self.numObjects):
                            for l in range(self.numLabels):
                                for m in range(self.numObjects):
                                    for n in range(self.numLabels):
                                        self.support[i,j] += self.strength[k,l]*self.strength[m,n]*self.compatibility[i,j,k,l,m,n]
                    self.normalizeSupport(i)

    def normalizeSupport(self, i=None):
        if i is None:
            if self.normalizeSupportAcrossEntireMatrix:
                minimumSupport = np.amin(self.support)
                maximumSupport = np.amax(self.support)
                maximumSupport -= minimumSupport
                if maximumSupport == 0:
                    self.support.fill(0.0)
                else:
                    self.support = (self.support - minimumSupport)/maximumSupport
            else:
                if self.useMatrixMultiplication:
                    minimumSupport = np.amin(self.support, axis=1)
                    maximumSupport = np.amax(self.support, axis=1)
                    maximumSupport -= minimumSupport
                    shifted = (self.support.T - minimumSupport).T
                    self.support = np.divide(
                        shifted,
                        maximumSupport[:, None],
                        out=np.zeros_like(shifted),
                        where=maximumSupport[:, None] != 0,
                    )
                else:
                    for i in range(self.numObjects):
                        self.normalizeSupport(i)
        else:
            minimumSupport = np.amin(self.support[i,:])
            maximumSupport = np.amax(self.support[i,:])
            maximumSupport -= minimumSupport
            if maximumSupport == 0:
                self.support[i, :].fill(0.0)
            else:
                self.support[i, :] = (self.support[i, :] - minimumSupport)/maximumSupport

    def updateStrength(self):
        technique = 2
        if technique == 1:
            if self.useMatrixMultiplication:
                self.strength = self.strength + self.support*self.supportFactor
                self.normalizeStrength()
            else:
                for i in range(self.numObjects):
                    for j in range(self.numLabels):
                        self.strength[i,j] += self.support[i,j]*self.supportFactor
                    self.normalizeStrength(i)
        if technique == 2:
            if self.useMatrixMultiplication:
                tmp= self.strength + np.multiply(self.strength,self.support)
                den = tmp.sum(axis=1)
                self.strength = np.divide(
                    tmp,
                    den[:, None],
                    out=self.strength.copy(),
                    where=den[:, None] != 0,
                )
                self.normalizeStrength()
            else:
                for i in range(self.numObjects):
                    den = 0.0
                    for j in range(0,self.numLabels):
                        den += self.strength[i,j]*(1.0+self.support[i,j])
                    for j in range(0,self.numLabels):
                        if den != 0:
                            self.strength[i,j] = self.strength[i,j]*(1.0+self.support[i,j])/den
                    self.normalizeStrength(i)

    def normalizeStrength(self, i=None):
        technique = 2
        if i is None:
            if technique == 1 or technique == 2:
                minStrength = np.amin(self.strength, axis=1)
                self.strength = (self.strength.T - minStrength).T
                if technique == 2:
                    sumStrength = np.sum(self.strength, axis=1)
                    uniform = np.full_like(self.strength, 1.0 / self.numLabels)
                    self.strength = np.divide(
                        self.strength,
                        sumStrength[:, None],
                        out=uniform,
                        where=sumStrength[:, None] != 0,
                    )
        else:
            if technique == 1 or technique == 2:
                minStrength = np.amin(self.strength[i, :])
                self.strength[i, :] -= minStrength
                if technique == 2:
                    sumStrength = np.sum(self.strength[i, :])
                    if sumStrength == 0:
                        self.strength[i, :].fill(1.0 / self.numLabels)
                    else:
                        self.strength[i, :] /= sumStrength

    def iterate(self):
        print("iteration {}".format(self.iteration))
        self.updateSupport()
        self.updateStrength()
        self.iteration += 1
        if self.printMatrices:
            print('Support iter #',self.iteration)
            print(self.support)
            print('Strength iter #',self.iteration)
            print(self.strength)

    def assign(self):
        print('labeling from strength')
        self.objectToLabelMapping = np.zeros((self.numObjects,1))
        for i in range(0,self.numObjects):
            jmax = np.argmax(self.strength[i,:])
            self.objectToLabelMapping[i] = jmax
            print('Obj#',i,' Label# ',jmax,'strength ',self.strength[i,jmax])
            if False:
                if np.linalg.norm(self.objects[i,:] - self.labels[jmax,:]) > 1e-4:
                    print('strengths for object i',i)
                    print(self.strength[i,:])

    def saveCompatibilityForPlotting(self, compatibilityFilename, compatibility):
        compatibilityFile = open(compatibilityFilename, 'w')
        compatibilityText = ''
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One column in header row:
                compatibilityText += ',' + str(i) + str(j)  
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One row in header column:
                compatibilityText += '\n' + str(i) + str(j) 
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        # One compatibility value:
                        compatibilityText += ',' + str(compatibility[i,j,k,l])
        compatibilityFile.write(compatibilityText)
        compatibilityFile.close()
        
    def saveCompatibility(self, compatibilityFilename, compatibility):
        compatibilityFile = open(compatibilityFilename, 'w')
        compatibilityText = ''
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One column in header row:
                compatibilityText += ',' + '[' + str(i) + ']' + '[' + str(j) + ']'
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One row in header column:
                compatibilityText += '\n' + '[' + str(i) + ']' + '[' + str(j) + ']'
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        # One compatibility value:
                        compatibilityText += ',' + str(compatibility[i,j,k,l])
        compatibilityFile.write(compatibilityText)
        compatibilityFile.close()

    def main(self):
        self.initStrengthAndSupport()
        print('Num objects', self.numObjects)
        print('Num labels', self.numLabels)
        for i in range(self.iterations):
            self.iterate()
        print('support', self.support)
        print('strength', self.strength)
        self.assign()
