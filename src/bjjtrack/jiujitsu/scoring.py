BUFFER_SIZE = 10
TAKEDOWN_BUFFER = 6

class BJJudge():
    def __init__(self, timeout = 90, verbose = False, player1 = 'blue', player2 = 'red'):
        self.points1 = 0
        self.points2 = 0
        self.position = 'start'
        self.top = -1
        
        # 90 frames at 30fps = 3s
        self.TIMEOUT = timeout - BUFFER_SIZE

        #FLAGS
        #standing
        self.standing = False
        #guard pass
        self.pass1 = False
        self.pass2 = False
        self.guard_established = False
        #reversal
        self.sweep1 = False
        self.sweep2 = False
        self.takedown1 = False
        self.takedown2 = False
        #mount | back
        self.four_points1 = False
        self.four_points2 = False
        self.verbose = verbose
        self.player1 = player1
        self.player2 = player2
        self.buffer = []
        self.trace = []


    def reset_flags(self):
        #FLAGS
        #guard pass
        self.guard_established = False
        self.pass1 = False
        self.pass2 = False
        #reversal
        self.sweep1 = False
        self.sweep2 = False
        self.takedown2 = False
        self.takedown1 = False
        #mount | back
        self.four_points2 = False
        self.four_points1 = False


    def get_top(self, position):
        if 'transition' in position:
            return self.top

        if 'back' in position:
            return -1 #self.top

        if 'guard' in position or 'takedown' in position:
            if '1' in position:
                return 2
            elif '2' in position:
                return 1
            else:
                return self.top
                
        if '1' in position:
            return 1
        elif '2' in position:
            return 2
        else:
            return -1
    

    def dominant_position(self, position):
        return ('mount' in position or 'back' in position or 'side' in position)


    def bottom_position(self, position):
        return 'turtle' in position or 'guard' in position


    def update_flags(self, new_position, top, frame):
        #raise guard pass flag
        if ('side' in new_position or 'mount' in new_position) and 'guard' in self.position and self.guard_established:
            if '1' in new_position and not '1' in self.position:
                self.pass1 = frame
            if '2' in new_position and not '2' in self.position:
                self.pass2 = frame
            self.guard_established = False

        #COMBINE SWEEP AND TAKEDOWN
        #raise and kill takedown flag
        if 'takedown' in new_position:
            self.reset_flags()
            if '1' in new_position:
                self.takedown1 = frame
                self.takedown2 = False
            if '2' in new_position:
                self.takedown2 = frame
                self.takedown1 = False

        #raise sweep flag
        elif (self.top != -1 and self.top != top and
            self.bottom_position(self.position)):
            if top == 1 and not self.takedown1:
                if self.sweep2:
                    self.sweep2 = False
                else:
                    self.sweep1 = frame
            if top == 2 and not self.takedown2:
                if self.sweep1:
                    self.sweep1 = False
                else:
                    self.sweep2 = frame


        #raise mount | back flag
        if 'mount' in new_position or 'back' in new_position:
            if '1' in new_position:
                self.four_points1 = frame
                if 'back' in new_position:
                    self.sweep1 = False
            if '2' in new_position:
                self.four_points2 = frame
                if 'back' in new_position:
                    self.sweep2 = False

        
        #kill pass flag
        if self.bottom_position(new_position):
            self.pass1 = False
            self.pass2 = False

        
        if 'standing' in new_position:
            self.standing = frame
        else:
            self.standing = False
        
        if 'guard' in new_position:
            self.guard = frame
        else:
            self.guard = False
        
        self.top = top


    def evaluate_flags(self, new_position, top, frame):
        """Evaluate the pass, sweep, takedown mount and back flags 
            based on bjj rules."""
        if new_position == 'transition':
            return
        #check for standing
        if new_position == 'standing':
            #check if the standing timeout is passed
            if self.standing and self.standing + self.TIMEOUT < frame:
                #standing position is valid, reset flags
                self.reset_flags()
                self.standing = False
            else: #the athletes are standing, but no timeout - do nothing
                return
            
        #check if guard is established
        if 'guard' in new_position:
            if self.guard and self.guard + self.TIMEOUT < frame and not self.guard_established:
                print('Guard established by ?')
                self.guard_established = True

        #guard pass
        if self.pass1 and self.pass1 + self.TIMEOUT < frame:
            self.points1 += 3
            self.pass1 = False
            if self.verbose: print(f'{self.player1} passed - {frame} ({self.points1} : {self.points2})')
            self.trace.append([frame,3,0])
        if self.pass2 and self.pass2 + self.TIMEOUT < frame:
            self.points2 += 3
            self.pass2 = False
            if self.verbose: print(f'{self.player2} passed - {frame} ({self.points1} : {self.points2})')
            self.trace.append([frame, 3,1])
        #sweep
        if self.sweep1 and self.sweep1 + self.TIMEOUT < frame:
            self.points1 += 2
            self.sweep1 = False
            if self.verbose: print(f'{self.player1} swept - {frame} ({self.points1} : {self.points2})')
            self.trace.append([frame, 2,0])
        if self.sweep2 and self.sweep2 + self.TIMEOUT < frame:
            self.points2 += 2
            self.sweep2 = False
            if self.verbose: print(f'{self.player2} swept - {frame} ({self.points1} : {self.points2})')
            self.trace.append([frame, 2,1])
        #takedown
        if self.takedown1:
            if new_position == 'takedown1':
                self.takedown1 = frame
            elif self.takedown1 + self.TIMEOUT < frame and top == 1:
                self.points1 += 2
                if self.verbose: print(f'{self.player1} took down {self.player2} - {frame} ({self.points1} : {self.points2})')
                self.trace.append([frame, 2,0])
                self.takedown1 = False
            elif top == 2:
                self.takedown1 = False

        if self.takedown2:
            if new_position == 'takedown2':
                self.takedown2 = frame
            elif self.takedown2 + self.TIMEOUT < frame and top == 2:
                self.points2 += 2
                if self.verbose: print(f'{self.player2} took down {self.player1} - {frame} ({self.points1} : {self.points2})')
                self.trace.append([frame, 2, 1])
                self.takedown2 = False
            elif top == 1:
                self.takedown2 = False

        #mount | back
        if (self.four_points1 and self.four_points1  +self.TIMEOUT < frame and 
           ('back1' in new_position or 'mount1' in new_position)):

            self.points1 += 4
            if 'mount' in new_position:
                if self.verbose: print(f'{self.player1} mounted {self.player2} - {frame} ({self.points1} : {self.points2})')
                self.trace.append([frame, 4, 0])
            else:
                if self.verbose: print(f"{self.player1} took {self.player2}'s back - {frame} ({self.points1} : {self.points2})")
                self.trace.append([frame, 4, 0])
            self.four_points1 = False

        if (self.four_points2 and self.four_points2 + self.TIMEOUT < frame and 
           ('back2' in new_position or 'mount2' in new_position)):

            self.points2 += 4
            if 'mount' in new_position:
                if self.verbose: print(f'{self.player2} mounted {self.player1} - {frame} ({self.points1} : {self.points2})')
                self.trace.append([frame, 4, 1])
            else:
                if self.verbose: print(f"{self.player2} took {self.player1}'s back - {frame} ({self.points1} : {self.points2})")
                self.trace.append([frame, 4, 1])
            self.four_points2 = False

       
    def update(self, new_position, frame):
        top = self.get_top(new_position)
        if new_position == self.position:
            if len(self.buffer) > 0:
                print(f'Buffer: {self.buffer}')
            self.buffer = []
        if  new_position != self.position and new_position != 'transition':
            # TODO: refactor this?
            if len(self.buffer) < BUFFER_SIZE and not ('takedown' in new_position and len(self.buffer) >= TAKEDOWN_BUFFER):
                top = self.top
                if (len(self.buffer) == 0 or new_position == self.buffer[0]):
                    self.buffer.append(new_position)
                else:
                    self.buffer = []
                new_position = self.position
            else:
                self.buffer = []
                self.update_flags(new_position, top, frame)
                if self.verbose: print(f'Changing from {self.position :<14} to { new_position:<14}: {frame}')
                self.position =  new_position
        self.evaluate_flags(new_position, top, frame)
        self.top = top
        

    def get_result(self):
        if self.points1 > self.points2:
            if self.verbose: print(f"{self.player1} won")
        elif self.points1 == self.points2:
            if self.verbose: print("Draw")
        else:
            if self.verbose: print(f"{self.player2} won")
        return self.points1, self.points2

    def get_trace(self):
        return self.trace