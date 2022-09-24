import pygame
from pygame.locals import *
import sys
import time
import random

# 950 x 500

'''f=open("results.txt",'a')
f.write("TYPE HISTORY RESULTS")
f.write("\n")
f.close()'''
    
class Game:
       
    def __init__(self):
        self.w=750
        self.h=500
        self.reset=True
        self.active = False
        self.input_text=''
        self.word = ''
        self.time_start = 0
        self.total_time = 0
        self.accuracy = '0%'
        self.results = 'Time:0 Accuracy:0 % Wpm:0 '
        self.wpm = 0
        self.end = False
        self.HEAD_C = (240,240,240)
        self.TEXT_C = (0,102,255)
        
        
        pygame.init()
        self.open_img = pygame.image.load('typetest.jpg')
        self.open_img = pygame.transform.scale(self.open_img, (self.w,self.h))

        self.bg = pygame.image.load('background.jpg')
        self.bg = pygame.transform.scale(self.bg, (800,650))

        self.screen = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('TEST-YOUR-TYPING-SKILLS')
       

    def check(self, string, sub_str):
        if not(string.startswith(sub_str)):
            return False
        else:
            return True
        
    def draw_text(self, screen, msg, y ,fsize, color):
        font = pygame.font.Font(None, fsize)

        # create a text surface object,
        # on which text is drawn on it.
        text = font.render(msg, 1,color)
        # create a rectangular object for the
        # text surface object
        text_rect = text.get_rect(center=(self.w/2, y))
        # copying the text surface object
        # to the screen surface object
        # at the center coordinate.        
        screen.blit(text, text_rect)
        # Draws the surface object to the screen.
        pygame.display.update()   
        
    def get_sentence(self):
        f = open('sentences.txt').read()
        sentences = f.split('\n')
        sentence = random.choice(sentences)
        prev=str(sentence)
        while 1 :
         if prev != sentence :
          return sentence
         else :
             prev=sentence
             sentence=random.choice(sentences)
    def show_results(self, screen):
        if(not self.end):
            #Calculate time
            self.total_time = time.time() - self.time_start
            
               
            #Calculate accuracy
            count = 0
            noofwords=(self.word).split()
            textbreak=(self.input_text).split()
            
            
            for i,c in enumerate(textbreak):
                
                try:
                    if noofwords[i] == c:
                        count += 1
                        
                except:
                    pass
                
            
            self.accuracy = count/len(noofwords)*100
            
            
            #Calculate words per minute
            self.wpm = count*60/(self.total_time)
            self.end = True
            #print(self.total_time)
            self.text=""
            if round(self.accuracy) < 50 :
                self.text+='ABYSMAL'
            elif round(self.accuracy) < 70 :
                self.text+='GOOD'
            else :
                self.text+='EXCELLENT'
                
            self.results = 'Time:'+str(round(self.total_time)) +" secs   Accuracy:"+ str(round(self.accuracy)) + "%" + '   Wpm: ' + str(round(self.wpm)) + "   " + self.text
            if(self.accuracy > 75):
                self.RESULT_C = (0,255,0)
            elif(self.accuracy > 50):
                self.RESULT_C = (255,235,0)
            else:
                self.RESULT_C = (255,0,0)
                
            f=open('results.txt','a')
            f.write(self.results)
            f.write("\n")
            f.close()
            
            self.screen.fill((0,0,0), (337,418,75,25))
            self.draw_text(screen,"Reset", self.h - 70, 26, (155,155,155))
            
            #print(self.results)
            pygame.display.update()


    def run(self):
        self.reset_game()
    
       
        self.running=True
        while(self.running):
            clock = pygame.time.Clock()
            self.screen.fill((0,0,0), (50,250,650,50))
            pygame.draw.rect(self.screen,self.HEAD_C, (50,250,650,50), 2)
            # update the text of user input
            
            if( self.check(self.word,self.input_text)== True):
                self.draw_text(self.screen, self.input_text, 274, 26,(0,250,0))
            elif( self.check(self.word,self.input_text) == False):
                self.draw_text(self.screen, self.input_text, 274, 26,(250,0,0))
                
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    pygame.display.quit()
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    x,y = pygame.mouse.get_pos()
                    #position of input box
                    if(x>=50 and x<=650 and y>=250 and y<=300):
                        self.active = True
                        self.input_text = ''
                        self.time_start = time.time()
                        
                    
                     # position of reset box
                    if(x>=310 and x<=510 and y>=390 and self.end):
                        self.reset_game()
                        x,y = pygame.mouse.get_pos()
         
                        
                elif event.type == pygame.KEYDOWN:
                    if self.active and not self.end:
                        if event.key == pygame.K_RETURN:
                            
                            self.show_results(self.screen)
                            
                            self.draw_text(self.screen, self.results,350, 28, self.RESULT_C)  
                            self.end = True
                            
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        else:
                            try:
                                self.input_text += event.unicode
                            except:
                                pass
            
            pygame.display.update()
             
        '''clock.tick() specifies how fast you want to change the game display in other words how fast the loop runs'''
        clock.tick(120)

    def reset_game(self):
        self.screen.blit(self.open_img, (0,0))

        pygame.display.update()
        time.sleep(1)
        
        self.reset=False
        self.end = False

        self.input_text=''
        self.word = ''
        self.time_start = 0
        self.total_time = 0
        self.wpm = 0

        # Get random sentence 
        self.word = self.get_sentence()
        if (not self.word): self.reset_game()
        #drawing heading
        self.screen.fill((0,0,0))
        self.screen.blit(self.bg,(0,0))
        msg = "TEST-YOUR-TYPING-SKILLS"
        self.draw_text(self.screen, msg,60, 60,self.HEAD_C)  
        # draw the rectangle for input box
        pygame.draw.rect(self.screen,(255,192,25), (50,250,650,50), 2)

        # draw the sentence string
        self.draw_text(self.screen, self.word,200, 28,self.TEXT_C)
        
        pygame.display.update()



Game().run()

