import matplotlib.pyplot as plt
import math

# Create a list of colors for each index
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray', 'pink', 'olive', 'cyan', 'magenta', 'teal', 'navy', 'maroon', 'gold']

def reverse_star_transform(new_stars, img_width, img_height, elevation, angle):
    original_stars = []
    
    for new_star in new_stars:
        new_x, new_y = new_star
        r = math.sqrt((new_x - img_width/2)**2 + (new_y - img_height/2)**2)
        theta = math.atan2(new_y - img_height/2, new_x - img_width/2) - math.radians(angle)
        
        x = r * math.cos(theta)
        y = r * math.sin(theta) / elevation
        
        star_x = x + img_width/2
        star_y = y + img_height/2

        # Calculate the center of the image
        center_x = int(img_width/2)
        center_y = int(img_height/2)

        # Calculate the distance between the center and each coordinate
        distances = (star_x - center_x, star_y - center_y)
        
        # Mirror the distances across the y-axis
        mirrored_distances = (-distances[0], distances[1])
        

        star_x,star_y = (center_x + mirrored_distances[0], center_y + mirrored_distances[1]) 


        original_stars.append((star_x, star_y))
    
    return original_stars

sorted_list2 = [(643.0, 461.5), (1218.0, 416.0), (775.5, 401.0), (484.5, 392.5), (1015.0, 364.0), (1240.0, 334.5), (1062.5, 244.5)]
#initial [(1276.0, 462.0), (701.0, 415.5), (1143.5, 400.5), (1435.5, 391.5), (904.5, 363.5), (678.5, 333.5), (856.5, 243.0)]
#mirror [(644.0, 618.0), (1219.0, 664.5), (776.5, 679.5), (484.5, 688.5), (1015.5, 716.5), (1241.5, 746.5), (1063.5, 837.0)]
#sorted_list2= [(782.5, 1023.5), (783.5, 1158.5), (642.5, 1231.5), (423.0, 1362.0), (580.5, 1406.5), (418.5, 1517.0), (576.0, 1580.0)]

sorted_list1 = reverse_star_transform(sorted_list2, 512, 512, 1, 360-135 )

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first list with assigned colors
ax1.scatter([item[0] for item in sorted_list1], [item[1] for item in sorted_list1], c=[colors[i] for i in range(len(sorted_list1))])

# Set the title and labels for the first plot
ax1.set_title('URSA from Frame')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Plot the second list with assigned colors
ax2.scatter([item[0] for item in sorted_list2], [item[1] for item in sorted_list2], c=[colors[i] for i in range(len(sorted_list2))])

# Set the title and labels for the second plot
ax2.set_title('URSA from Known Position')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Show the plot
plt.show()
