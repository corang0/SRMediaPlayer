   M S B A M L   `   `   `  ���� H  DMyMediaPlayer, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null-clr-namespace:MyMediaPlayerMyMediaPlayer      MyMediaPlayer.MainWindow   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     V RMaterialDesignThemes.Wpf, Version=3.1.1.1361, Culture=neutral, PublicKeyToken=nullFmaterialDesign1http://materialdesigninxaml.net/winfx/xaml/themes  'localclr-namespace:MyMediaPlayer  -     � 
ResizeMode$  CanResizeWithGrip=� � Title$ 
MainWindow��$	��120��$
��1100�����	#FF494949.��� .+�� -    �Name$ grid4��$	��850��.+��� -    ��Name$ image��$
��Left=�$��	0,70,0,70q�$	��Top=�$��0��$y�Visible=��� -   $ Dimage��$
��Left=�$��	0,70,0,70q�$	��Top=�$��0��$y�Visible=�� -   $ grid5�� � ColumnDefinitions �� -    �Name$
 col1���� $��2 ��� -   $
 col2��.+��� -    ��Name$ splitter��
���   �$��1��$��2��$��Center=�$��Stretch=�$��20��$y�Hidden=�� -	   $ grid3��$��Bottom=�$
��Left=� � ShowGridLines . $	��950��$��50�� �� .+��� -
    ��Name$ Play_Button��$��
55,-20,0,0q�$
��Left=�   MaterialDesignIconButton#��[  	 � ToolTip$	 MaterialDesignIconButton�� ( $MaterialDesignTextFieldBoxBackground#���  .��'  !MaterialDesignThemes.Wpf.PackIcon  
   Kind$

 Play=�� -   $
 grid��$��Bottom=�$
��left=� . $	��950��$��50��$��0,0,0,0q�.+��� -    ��Name$ 
start_Time��$��00:00:00��$��8(�
�������$
��Left=�$��Center=�$��43��$��25��$��	3,-10,0,0q��� -    ��Name$ slider��$��26��$��
85,2,70,15q�$��Center=�$	��707���� -   $ finish_Time��$��00:00:00��$��8(�
�������$��Right=�$��0,-10,60,10q�$��Center=�$��43��$��25��$��1���� -   $ btnAudio�� / +MaterialDesignFloatingActionMiniLightButton#��[ $1	 +MaterialDesignFloatingActionMiniLightButton��$��902,-20,0,0q����	#FF484849
������ ���	#FFE4E2E6.��  $
 Audio=�$��24��$��24�����	#FF673AB7�� -   $ AudioVolumn��$y�Hidden=�
������� �� TickFrequency$ 5Q� �� TickPlacement$ BottomRight=� �� Orientation$ Vertical=� � Minimum$ 1Q� � Maximum$ 50Q�$��Right=�$��67��$��0,-30,-30,0q��� -    ��Name$ FilledCombo��   MaterialDesignFilledComboBox#��[ )  #MaterialDesignThemes.Wpf.HintAssist   Hint$
 MODE��$��80��$��60��$��0,-19,-129,9q�$��Right=�
�������$
��Bold"�$��13(�{���  �� Path$ 	IsChecked� �� ElementName$1 +MaterialDesignFilledComboBoxEnabledComboBox��.@��� $��FSRCNN�����	#FF404042�� $
��GMFN�����	#FF404042